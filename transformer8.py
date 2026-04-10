#Transformer_base_calling images to letter bases for 454.Bio.  JMR April 5 2023

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

def _build_and_train_transformer(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors):
    """Build a new transformer model, train it on the current images, and return (model, window_size)."""
    X_train, y_train = create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)
    input_layer = Input(shape=(window_size, 4))
    # d_model=4 requires num_heads * key_dim == 4 (e.g. 2 heads × 2)
    x = MultiHeadAttention(num_heads=2, key_dim=2)(input_layer, input_layer)
    x = LayerNormalization(epsilon=1e-6)(Add()([input_layer, x]))
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
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
    print("transformer_base_calling")
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

    if auto_train:
        # Train fresh without prompts (used by "run all" mode)
        transformer_model = _build_and_train_transformer(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)
    elif model_files:
        print("Available models:")
        for i, f in enumerate(model_files):
            print(f"{i + 1}. {f}")

        choice = input("Enter the number of the model to load, or name (must have letters/without extension) to create a new one: ")
        if not choice.isdigit():
            save_choice = input("Do you want to save this model (hit return to skip): ")
            if save_choice:
                filename = choice

            transformer_model = _build_and_train_transformer(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)

            if save_choice:
                transformer_model.save(os.path.join(folder_path, f"{filename}_transformer_model_w{window_size}.h5"))
        else:
            file_index = int(choice) - 1
            # Load the chosen model
            transformer_model = load_model(os.path.join(folder_path,model_files[file_index]))
            saved_window_size = extract_window_size(model_files[file_index])
            # Set the window size to the saved model's window size
            if saved_window_size is not None:
                window_size = saved_window_size
    else:
        print("No saved models found.")
        save_choice = input("Do you want to save this model (hit return to skip): ")
        if save_choice:
            filename = input("Enter a filename for the model (without extension): ")

        transformer_model = _build_and_train_transformer(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)

        if save_choice:
            transformer_model.save(os.path.join(folder_path, f"{filename}_transformer_model_w{window_size}.h5"))
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
