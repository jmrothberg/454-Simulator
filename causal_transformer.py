# Autoregressive (GPT-style) Transformer base caller for 454.Bio.
# Uses CAUSAL masking: position t can only attend to cycles 0..t.
# Processes the FULL read in one forward pass — no sliding window.
# Each position predicts its base from all preceding signal.
# JMR / April 2025

import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, Add, Softmax,
    MultiHeadAttention, Embedding,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

_BASE_ORDER = ("A", "C", "G", "T")
_BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

cwd = os.getcwd()
folder_name = "Sim Models"
folder_path = os.path.join(cwd, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


def _sinusoidal_pos_encoding(max_len, d_model):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""
    pos = np.arange(max_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000.0, (2 * (i // 2)) / np.float64(d_model))
    angles = pos * angle_rates
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return angles.astype(np.float32)  # (max_len, d_model)


def _causal_transformer_block(x, d_model, num_heads, ff_dim, dropout_rate=0.2):
    """Single causal transformer block — attention is masked so position t
    can only attend to positions 0..t (past + self, no future)."""
    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
    )(x, x, use_causal_mask=True)
    attn = Dropout(dropout_rate)(attn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))

    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(d_model)(ffn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, ffn]))
    return x


def _base_from_vector(tpl, base_keys):
    arr = np.asarray(tpl, dtype=np.float64).ravel()[:4]
    return base_keys[int(np.argmax(arr))]


# ── Training data ──────────────────────────────────────────────────────

def _create_training_set(images, num_templates, templates, num_training_templates,
                         num_cycles, base_colors, image_dim):
    """Build (X, Y) where each sample is a FULL read (num_cycles, 4).
    X = noisy observed colors / 255,  Y = one-hot true bases."""
    base_keys = list(base_colors.keys())
    X, Y = [], []
    for idx in range(num_training_templates):
        row, col = divmod(idx, image_dim)
        seq = templates[idx]
        letters = "".join(_base_from_vector(tpl, base_keys) for tpl in seq)

        obs = np.zeros((num_cycles, 4), dtype=np.float64)
        tgt = np.zeros((num_cycles, 4), dtype=np.float64)
        for c in range(num_cycles):
            obs[c] = np.asarray(images[c][row][col], dtype=np.float64).ravel()[:4]
            if c < len(letters):
                tgt[c, _BASE_MAP[letters[c]]] = 1.0
            else:
                tgt[c, 0] = 1.0  # pad with A (won't affect accuracy past real length)
        X.append(obs / 255.0)
        Y.append(tgt)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ── Model ──────────────────────────────────────────────────────────────

def _build_causal_model(num_cycles, d_model=64, num_heads=4, ff_dim=128,
                        num_blocks=4, dropout_rate=0.2):
    """GPT-style autoregressive transformer for base calling.
    Input : (batch, num_cycles, 4) — noisy observation sequence
    Output: (batch, num_cycles, 4) — softmax base probabilities per position
    Causal mask ensures position t only sees cycles 0..t."""
    inp = Input(shape=(num_cycles, 4))
    x = Dense(d_model)(inp)  # project 4 → d_model

    # Add sinusoidal positional encoding as a constant
    pos_enc = _sinusoidal_pos_encoding(num_cycles, d_model)
    x = x + pos_enc[None, :, :]  # broadcast over batch

    for _ in range(num_blocks):
        x = _causal_transformer_block(x, d_model, num_heads, ff_dim, dropout_rate)

    out = Dense(4, activation="softmax")(x)  # per-position 4-class prediction

    model = Model(inputs=inp, outputs=out)
    return model


def _train_model(model, X, Y):
    stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)
    model.fit(X, Y, batch_size=32, epochs=100, validation_split=0.1,
              callbacks=[stop], verbose=1)


# ── Main entry point ───────────────────────────────────────────────────

def causal_transformer_base_calling(images, num_cycles, num_templates, templates,
                                    num_training_templates, window_size, auto_train=False):
    """Autoregressive (causal) transformer base caller.
    Processes the full read (all cycles) per template in one pass.
    Position t predicts its base using only observations from cycles 0..t.
    window_size arg is accepted for API compatibility but not used."""
    print("Autoregressive (Causal) Transformer base calling")

    base_colors = {
        "A": (1.0, 0.0, 0.0, 0.0),
        "C": (0.0, 1.0, 0.0, 0.0),
        "G": (0.0, 0.0, 1.0, 0.0),
        "T": (0.0, 0.0, 0.0, 1.0),
    }

    image_dim = math.ceil(math.sqrt(num_templates))
    _auto_ckpt = os.path.join(folder_path, f"_auto_causal_transformer_c{num_cycles}.h5")

    if auto_train:
        X_train, Y_train = _create_training_set(
            images, num_templates, templates, num_training_templates,
            num_cycles, base_colors, image_dim,
        )

        if os.path.exists(_auto_ckpt):
            print(f"Warm-starting from checkpoint: {os.path.basename(_auto_ckpt)}")
            model = load_model(_auto_ckpt, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.0003),
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])
        else:
            print("No checkpoint — building new causal transformer")
            model = _build_causal_model(num_cycles)
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])

        _train_model(model, X_train, Y_train)
        model.save(_auto_ckpt)
        print(f"Checkpoint saved: {os.path.basename(_auto_ckpt)}")

    else:
        # Interactive mode
        if os.path.exists(_auto_ckpt):
            print(f"Checkpoint found: {os.path.basename(_auto_ckpt)}")
            choice = input("Enter to load checkpoint, or 'new' to train fresh: ").strip()
            if choice == "":
                model = load_model(_auto_ckpt, compile=False)
                model.compile(optimizer=Adam(learning_rate=0.0003),
                              loss="categorical_crossentropy",
                              metrics=["accuracy"])
                print("Loaded checkpoint (no retraining)")
            else:
                X_train, Y_train = _create_training_set(
                    images, num_templates, templates, num_training_templates,
                    num_cycles, base_colors, image_dim,
                )
                model = _build_causal_model(num_cycles)
                model.compile(optimizer=Adam(learning_rate=0.001),
                              loss="categorical_crossentropy",
                              metrics=["accuracy"])
                _train_model(model, X_train, Y_train)
                model.save(_auto_ckpt)
                print(f"Checkpoint saved: {os.path.basename(_auto_ckpt)}")
        else:
            print("No checkpoint — building new causal transformer")
            X_train, Y_train = _create_training_set(
                images, num_templates, templates, num_training_templates,
                num_cycles, base_colors, image_dim,
            )
            model = _build_causal_model(num_cycles)
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])
            _train_model(model, X_train, Y_train)
            model.save(_auto_ckpt)
            print(f"Checkpoint saved: {os.path.basename(_auto_ckpt)}")

    # ── Batched inference on ALL templates ──
    called_bases = [""] * num_templates
    all_obs = np.zeros((num_templates, num_cycles, 4), dtype=np.float32)
    for idx in range(num_templates):
        row, col = divmod(idx, image_dim)
        for c in range(num_cycles):
            all_obs[idx, c] = np.asarray(images[c][row][col], dtype=np.float64).ravel()[:4]
    all_obs /= 255.0

    all_preds = model.predict(all_obs, batch_size=256, verbose=0)  # (N, num_cycles, 4)

    base_keys = list(base_colors.keys())
    for idx in range(num_templates):
        preds = all_preds[idx]  # (num_cycles, 4)
        seq = "".join(_BASE_ORDER[int(np.argmax(preds[c]))] for c in range(num_cycles))
        called_bases[idx] = seq

        true_seq = "".join(_base_from_vector(tpl, base_keys) for tpl in templates[idx])
        print(f"Template              {idx}: {true_seq}")
        print(f"Causal Transformer    {idx}: {seq}")

    return called_bases
