# Bidirectional Window Encoder ("BERT-style") base caller for 454.Bio.
# NOT autoregressive: uses a fixed-size window with bidirectional self-attention
# (every position sees every other position).  Predicts all positions in the
# window simultaneously and keeps the center position as the final call.
# JMR April 5 2023

import math
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, MultiHeadAttention, LayerNormalization, Add, Reshape, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

cwd = os.getcwd()

# Define the folder name
folder_name = 'Sim Models'

# Create the folder path
folder_path = os.path.join(cwd, folder_name)

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print ("Made New Folder ", folder_path)
_BASE_ORDER = ("A", "C", "G", "T")

def onehot_to_base(predicted_colors):
    """Argmax per 4-vector (matches softmax output; avoids dot vs corner tie quirks)."""
    predicted_base_seq = []
    flat = np.asarray(predicted_colors, dtype=np.float64).ravel()
    for i in range(0, len(flat), 4):
        block = flat[i : i + 4]
        predicted_base_seq.append(_BASE_ORDER[int(np.argmax(block))])
    return predicted_base_seq


def _train_transformer_model(model, X_train, y_train):
    stop = EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True, verbose=0
    )
    model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=80,
        validation_split=0.1,
        callbacks=[stop],
        verbose=1,
    )

def _base_from_vector(tpl, base_keys):
    arr = np.asarray(tpl, dtype=np.float64).ravel()[:4]
    return base_keys[int(np.argmax(arr))]

def create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors):
    print ("create_training_set")
    X_train, y_train = [], []
    base_keys = list(base_colors.keys())
    for training_seq_idx in range(num_training_templates):

        row, col = divmod(training_seq_idx, math.ceil(math.sqrt(num_templates)))
        training_seq = templates[training_seq_idx]

        letters = "".join(_base_from_vector(tpl, base_keys) for tpl in training_seq)
        base_combinations = [letters[i:i + window_size] for i in range(len(letters) - window_size + 1)]

        for i, base_combination in enumerate(base_combinations):
            cycle_offset = i
            spot_colors = [np.asarray(images[min(cycle_offset + j, num_cycles - 1)][row][col], dtype=np.float64).ravel()[:4] for j in range(window_size)]

            expected_colors = [np.array(base_colors[base]) for base in base_combination]

            # BUGFIX: training inputs must match inference (/255) or loss fights wrong scale.
            X_train.append(np.array(spot_colors, dtype=np.float64) / 255.0)
            y_train.append(np.array(expected_colors))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = y_train.reshape(y_train.shape[0], 4 * window_size)

    return X_train, y_train

def extract_window_size(model_filename):
    pattern = re.compile(r"_transformer_model_w(\d+)\.h5$")
    match = pattern.search(model_filename)
    if match:
        return int(match.group(1))
    else:
        return None

def _transformer_block(x, d_model, num_heads, ff_dim, dropout_rate=0.2):
    """Single transformer encoder block: MHA + residual + LayerNorm + FFN + residual + LayerNorm."""
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    attn = Dropout(dropout_rate)(attn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))
    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(d_model)(ffn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, ffn]))
    return x


def _build_and_train_transformer(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors):
    """Build a bidirectional window encoder (NOT autoregressive).
    Uses full self-attention within a fixed window — every position sees every other.
    Returns (model, window_size)."""
    X_train, y_train = create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)

    d_model = 64
    input_layer = Input(shape=(window_size, 4))
    x = Dense(d_model)(input_layer)  # project 4 channels → d_model

    # 3 transformer encoder blocks
    for _ in range(3):
        x = _transformer_block(x, d_model=d_model, num_heads=4, ff_dim=128, dropout_rate=0.2)

    x = Flatten()(x)
    output_layer = Dense(4 * window_size)(x)
    output_layer = Reshape((window_size, 4))(output_layer)
    output_layer = Softmax(axis=-1)(output_layer)
    output_layer = Flatten()(output_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')
    _train_transformer_model(model, X_train, y_train)
    return model


def transformer_base_calling(images, num_cycles, num_templates, templates, num_training_templates, window_size, auto_train=False):
    """Bidirectional Window Encoder — slides a fixed window across cycles,
    predicts all positions with full (non-causal) self-attention, keeps center."""
    print("Bidirectional Window Encoder base calling")
    base_colors = {
        'A': (1.0, 0.0, 0.0, 0.0),
        'C': (0.0, 1.0, 0.0, 0.0),
        'G': (0.0, 0.0, 1.0, 0.0),
        'T': (0.0, 0.0, 0.0, 1.0)
    }

    # Check for available saved models
    pattern = re.compile(r'_transformer_model_w\d+\.h5$')
    #model_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if pattern.search(f)]
    model_files = [f for f in os.listdir(folder_path) if pattern.search(f)]

    # Auto-checkpoint: always saved after training so the model improves across runs.
    _auto_ckpt = os.path.join(folder_path, f"_auto_transformer_w{window_size}.h5")

    if auto_train:
        X_train, y_train = create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)

        if os.path.exists(_auto_ckpt):
            print(f"Warm-starting from checkpoint: {os.path.basename(_auto_ckpt)}")
            transformer_model = load_model(_auto_ckpt, compile=False)
            transformer_model.compile(optimizer=Adam(learning_rate=0.0003), loss='categorical_crossentropy')
        else:
            print("No checkpoint found — building new transformer model")
            transformer_model = _build_and_train_transformer(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)
            transformer_model.save(_auto_ckpt)
            print(f"Checkpoint saved: {os.path.basename(_auto_ckpt)}")
            # Model already trained inside _build_and_train — skip to inference
            X_train = None

        if X_train is not None:
            _train_transformer_model(transformer_model, X_train, y_train)
            transformer_model.save(_auto_ckpt)
            print(f"Checkpoint saved: {os.path.basename(_auto_ckpt)}")

    elif os.path.exists(_auto_ckpt):
        # Checkpoint exists — default to loading it (just hit Enter)
        print(f"Checkpoint found: {os.path.basename(_auto_ckpt)}")
        if model_files:
            print("Other saved models:")
            for i, f in enumerate(model_files):
                print(f"  {i + 1}. {f}")
        choice = input("Enter to load checkpoint, model # to load another, or 'new' to train fresh: ").strip()
        if choice == "" :
            print(f"Loading checkpoint: {os.path.basename(_auto_ckpt)}")
            transformer_model = load_model(_auto_ckpt, compile=False)
        elif choice.isdigit() and model_files:
            file_index = int(choice) - 1
            transformer_model = load_model(os.path.join(folder_path, model_files[file_index]))
            saved_window_size = extract_window_size(model_files[file_index])
            if saved_window_size is not None:
                window_size = saved_window_size
        else:
            transformer_model = _build_and_train_transformer(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)
            transformer_model.save(_auto_ckpt)
    else:
        print("No checkpoint or saved models found — training new model.")
        transformer_model = _build_and_train_transformer(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)
        transformer_model.save(_auto_ckpt)
    # Batched inference: collect ALL windows across ALL templates, predict once.
    called_bases = [''] * num_templates
    middle_index = window_size // 2
    image_dim = math.ceil(math.sqrt(num_templates))
    cycle_start = -middle_index
    cycle_end = num_cycles - window_size + middle_index + 1
    n_windows = cycle_end - cycle_start

    all_windows = []
    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, image_dim)
        for cycle in range(cycle_start, cycle_end):
            window = np.zeros((window_size, 4), dtype=np.float64)
            for w in range(window_size):
                c = cycle + w
                if 0 <= c < num_cycles:
                    window[w] = np.asarray(images[c][row][col], dtype=np.float64).ravel()[:4]
            all_windows.append(window)

    all_X = np.array(all_windows, dtype=np.float64) / 255.0
    all_preds = transformer_model.predict(all_X, batch_size=256, verbose=0)

    idx = 0
    for seq_idx in range(num_templates):
        chars = []
        for _ in range(n_windows):
            pred = all_preds[idx]
            flat = np.asarray(pred, dtype=np.float64).ravel()
            chars.append(_BASE_ORDER[int(np.argmax(flat[middle_index * 4 : middle_index * 4 + 4]))])
            idx += 1
        called_bases[seq_idx] = ''.join(chars)
        print("Template         ", seq_idx, ":", ''.join([base for v in templates[seq_idx] for base, color in base_colors.items() if np.array_equal([val / 255.0 for val in v], color)]))
        print("Transformer Called bases ", seq_idx, ":", called_bases[seq_idx])

    return called_bases
