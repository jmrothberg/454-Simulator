# CNN base caller.
import math
import os
import pickle
import re

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.models import load_model

# Training uses inputs in [0,1]; legacy pickles use scale 1.0 (raw ~255) at inference.
_INPUT_SCALE_KEY = 255.0

def train_cnn_classifier(X_train, onehot_encoded, window_size, iterations):
    input_shape = (window_size, 4)

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(onehot_encoded.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    n = max(1, int(iterations))
    model.fit(
        X_train.reshape(-1, window_size, 4),
        onehot_encoded,
        epochs=n,
        batch_size=32,
        verbose=1,
    )

    return model


def _base_from_vector(tpl, base_keys):
    arr = np.asarray(tpl, dtype=np.float64).ravel()[:4]
    return base_keys[int(np.argmax(arr))]

def create_training_set_cnn(images, num_templates, templates, num_training_templates, window_size, num_cycles,
                            base_colors):
    x, y = [], []
    base_keys = list(base_colors.keys())

    for training_seq_idx in range(num_training_templates):
        row, col = divmod(training_seq_idx, math.ceil(math.sqrt(num_templates)))
        training_seq = templates[training_seq_idx]

        letters = "".join(_base_from_vector(tpl, base_keys) for tpl in training_seq)
        base_combinations = [letters[i:i + window_size] for i in range(len(letters) - window_size + 1)]

        for i, base_combination in enumerate(base_combinations):
            cycle_offset = i
            spot_colors = [np.asarray(images[min(cycle_offset + j, num_cycles - 1)][row][col], dtype=np.float32).ravel()[:4] for j in range(window_size)]

            x.append(np.array(spot_colors, dtype=np.float32) / 255.0)
            y.append(base_combination)

    return np.array(x, dtype=np.float32), np.array(y)


def base_calling_cnn(images, num_cycles, num_templates, templates, num_training_templates, window_size, iterations, auto_train=False):
    print("base_calling_cnn")
    base_colors = {
        'A': (255.0, 0.0, 0.0, 0.0),
        'C': (0.0, 255.0, 0.0, 0.0),
        'G': (0.0, 0.0, 255.0, 0.0),
        'T': (0.0, 0.0, 0.0, 255.0)
    }

    # Get the current working directory
    cwd = os.getcwd()
    print("Current working directory: ", cwd)
    # Define the folder name
    folder_name = 'Sim Models'

    # Create the folder path
    folder_path = os.path.join(cwd, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Made New Folder ", folder_path)
    # Check for available saved training sets
    pattern = re.compile(r'_cnn_classifier_w(\d+)_i(\d+)\.pkl$')

    # training_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if pattern.search(f)]
    training_files = [f for f in os.listdir(folder_path) if pattern.search(f)]

    cnn_input_scale = _INPUT_SCALE_KEY

    def _train_fresh():
        x_tr, y_tr = create_training_set_cnn(images, num_templates, templates, num_training_templates,
                                              window_size, num_cycles, base_colors)
        le = LabelEncoder()
        ohe = OneHotEncoder(sparse_output=False)
        ie = le.fit_transform(y_tr).reshape(-1, 1)
        oh = ohe.fit_transform(ie)
        model = train_cnn_classifier(x_tr, oh, window_size, iterations)
        return model, le, ohe

    if auto_train:
        # Train fresh without prompts (used by "run all" mode)
        cnn, label_encoder, onehot_encoder = _train_fresh()
    elif training_files:
        print("Available training sets:")
        for i, f in enumerate(training_files):
            print(f"{i + 1}. {f}")

        choice = input("Enter the number of the training set to load, or type 'new' to create a new one: ")
        if choice.lower() == 'new':
            cnn, label_encoder, onehot_encoder = _train_fresh()

            save_choice = input("Do you want to save this training set for later use? (y/n): ")
            if save_choice.lower() == 'y':
                filename = input("Enter a filename for the training set (without extension): ")
                cnn.save(os.path.join(folder_path, f"{filename}_cnn_classifier_w{window_size}_i{iterations}.h5"))
                with open(os.path.join(folder_path, f"{filename}_cnn_classifier_w{window_size}_i{iterations}.pkl"),
                          'wb') as f:
                    pickle.dump(
                        (label_encoder, onehot_encoder, window_size, iterations, _INPUT_SCALE_KEY), f
                    )

        else:
            file_index = int(choice) - 1
            training_file = training_files[file_index]
            with open(os.path.join(folder_path, f"{training_file}"), 'rb') as f:
                cnn = load_model(os.path.join(folder_path, f"{os.path.splitext(training_file)[0]}.h5"))
                meta = pickle.load(f)
                if len(meta) == 5:
                    label_encoder, onehot_encoder, window_size, iterations, cnn_input_scale = meta
                else:
                    label_encoder, onehot_encoder, window_size, iterations = meta
                    cnn_input_scale = 1.0

    else:
        print("No saved training sets found.")
        cnn, label_encoder, onehot_encoder = _train_fresh()

        save_choice = input("Do you want to save this training set for later use? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Enter a filename for the training set (without extension): ")
            # BUGFIX: match the other branch — weights in .h5, encoders in .pkl (load path expects this).
            cnn.save(os.path.join(folder_path, f"{filename}_cnn_classifier_w{window_size}_i{iterations}.h5"))
            with open(os.path.join(folder_path, f"{filename}_cnn_classifier_w{window_size}_i{iterations}.pkl"),
                      'wb') as f:
                pickle.dump(
                    (label_encoder, onehot_encoder, window_size, iterations, _INPUT_SCALE_KEY), f
                )

    # Batched inference: collect ALL windows across ALL templates, predict once.
    called_bases = [''] * num_templates
    middle_index = window_size // 2
    image_dim = math.ceil(math.sqrt(num_templates))
    cycle_start = -middle_index
    cycle_end = num_cycles - window_size + middle_index + 1
    n_windows = cycle_end - cycle_start
    scale = max(float(cnn_input_scale), 1e-12)

    all_windows = []
    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, image_dim)
        for cycle in range(cycle_start, cycle_end):
            window = np.zeros((window_size, 4), dtype=np.float32)
            for w in range(window_size):
                c = cycle + w
                if 0 <= c < num_cycles:
                    window[w] = np.asarray(images[c][row][col], dtype=np.float32).ravel()[:4]
            all_windows.append(window)

    all_X = np.array(all_windows, dtype=np.float32) / scale
    all_preds = cnn.predict(all_X, batch_size=256, verbose=0)
    all_pred_indices = np.argmax(all_preds, axis=1)
    all_decoded = label_encoder.inverse_transform(all_pred_indices)

    idx = 0
    for seq_idx in range(num_templates):
        chars = []
        for _ in range(n_windows):
            chars.append(all_decoded[idx][middle_index])
            idx += 1
        called_bases[seq_idx] = ''.join(chars)
        print("Template         ", seq_idx, ":", ''.join(
            [base for v in templates[seq_idx] for base, color in base_colors.items() if np.array_equal(v, color)]))
        print("cnn Called Bases ", seq_idx, ":", called_bases[seq_idx])

    return called_bases
