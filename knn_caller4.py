#knn_caller with ability to save past training sets. April 5 2023
#Jonathan M. Rothberg 454 Bio For Simulated sequence.
import math
import os
import pickle
import re

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def _base_from_vector(tpl, base_keys):
    """BUGFIX: tpl may be ndarray — no .index; use argmax on first 4 channels."""
    arr = np.asarray(tpl, dtype=np.float64).ravel()[:4]
    return base_keys[int(np.argmax(arr))]

def create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors):
    X, y = [], []
    base_keys = list(base_colors.keys())

    for training_seq_idx in range(num_training_templates):
        row, col = divmod(training_seq_idx, math.ceil(math.sqrt(num_templates)))
        training_seq = templates[training_seq_idx]

        letters = "".join(_base_from_vector(tpl, base_keys) for tpl in training_seq)
        base_combinations = [letters[i:i + window_size] for i in range(len(letters) - window_size + 1)]

        for i, base_combination in enumerate(base_combinations):
            cycle_offset = i
            spot_colors = [np.asarray(images[min(cycle_offset + j, num_cycles - 1)][row][col], dtype=np.float64).ravel()[:4] for j in range(window_size)]

            X.append(np.array(spot_colors).flatten())
            y.append(base_combination)

    return np.array(X, dtype=np.float64), np.array(y)


def _fit_knn_scaled(X_train, y_train, k):
    """Scale features (per-dimension) + distance-weighted kNN — better when channel scales drift."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    k_eff = max(1, min(int(k), len(X_train)))
    knn = KNeighborsClassifier(
        n_neighbors=k_eff, weights="distance", algorithm="auto", metric="euclidean"
    )
    knn.fit(Xs, y_train)
    return scaler, knn


def base_calling_knn(images, num_cycles, num_templates, templates, num_training_templates, window_size, k, auto_train=False):
    print("base_calling_knn")
    base_colors = {
        'A': (255.0, 0.0, 0.0, 0.0),
        'C': (0.0, 255.0, 0.0, 0.0),
        'G': (0.0, 0.0, 255.0, 0.0),
        'T': (0.0, 0.0, 0.0, 255.0)
    }

    # Get the current working directory
    cwd = os.getcwd()
    print ("Current working directory: ", cwd)
    # Define the folder name
    folder_name = 'Sim Models'

    # Create the folder path
    folder_path = os.path.join(cwd, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Made New Folder ", folder_path)

    # Check for available saved training sets
    pattern = re.compile(r'_knn_classifier_w(\d+)_k(\d+)\.pkl$')

    # Create a list of file names that match the pattern in the folder
    #training_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if pattern.search(f)]
    training_files = [f for f in os.listdir(folder_path) if pattern.search(f)]

    scaler = None

    if auto_train:
        # Train fresh without prompts (used by "run all" mode)
        X_train, y_train = create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)
        scaler, knn = _fit_knn_scaled(X_train, y_train, k)
    elif training_files:
        print("Available training sets:")
        for i, f in enumerate(training_files):
            print(f"{i + 1}. {f}")

        choice = input("Enter the number of the training set to load, or type 'new' to create a new one: ")
        if choice.lower() == 'new':
            X_train, y_train = create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)

            scaler, knn = _fit_knn_scaled(X_train, y_train, k)

            save_choice = input("Do you want to save this training set for later use? (y/n): ")
            if save_choice.lower() == 'y':
                filename = input("Enter a filename for the training set (without extension): ")
                with open(os.path.join(folder_path, f"{filename}_knn_classifier_w{window_size}_k{k}.pkl"), 'wb') as f:
                    pickle.dump({"version": 2, "scaler": scaler, "knn": knn}, f)
        else:
            file_index = int(choice) - 1
            with open(os.path.join(folder_path, training_files[file_index]), 'rb') as f:
                loaded = pickle.load(f)
                match = pattern.search(training_files[file_index])
                window_size, k = int(match.group(1)), int(match.group(2))
                if isinstance(loaded, dict) and loaded.get("version") == 2:
                    scaler, knn = loaded["scaler"], loaded["knn"]
                else:
                    scaler, knn = None, loaded

    else:
        print("No saved training sets found.")
        X_train, y_train = create_training_set(images, num_templates, templates, num_training_templates, window_size,
                                               num_cycles, base_colors)

        scaler, knn = _fit_knn_scaled(X_train, y_train, k)

        save_choice = input("Do you want to save this training set for later use? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Enter a filename for the training set (without extension): ")
            with open(os.path.join(folder_path, f"{filename}_knn_classifier_w{window_size}_k{k}.pkl"), 'wb') as f:
                pickle.dump({"version": 2, "scaler": scaler, "knn": knn}, f)

    # Batched inference: collect all windows per template, predict in one call.
    called_bases = [''] * num_templates
    middle_index = window_size // 2
    image_dim = math.ceil(math.sqrt(num_templates))
    cycle_start = -middle_index
    cycle_end = num_cycles - window_size + middle_index + 1
    n_windows = cycle_end - cycle_start

    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, image_dim)

        all_windows = []
        for cycle in range(cycle_start, cycle_end):
            window = np.zeros((window_size, 4), dtype=np.float64)
            for w in range(window_size):
                c = cycle + w
                if 0 <= c < num_cycles:
                    window[w] = np.asarray(images[c][row][col], dtype=np.float64).ravel()[:4]
            all_windows.append(window.ravel())

        X = np.array(all_windows, dtype=np.float64)
        if scaler is not None:
            X = scaler.transform(X)
        preds = knn.predict(X)

        called_bases[seq_idx] = ''.join(p[middle_index] for p in preds)
        print("Template         ", seq_idx, ":", ''.join([base for v in templates[seq_idx] for base, color in base_colors.items() if np.array_equal(v, color)]))
        print("kNN Called Bases ", seq_idx, ":", called_bases[seq_idx])

    return called_bases