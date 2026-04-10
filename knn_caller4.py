#knn_caller with ability to save past training sets. April 5 2023
#Jonathan M. Rothberg 454 Bio For Simulated sequence.
import math
import os
import pickle
import re

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors):
    X, y = [], []

    for training_seq_idx in range(num_training_templates):
        row, col = divmod(training_seq_idx, math.ceil(math.sqrt(num_templates)))
        training_seq = templates[training_seq_idx]

        base_combinations = ["".join([list(base_colors.keys())[tpl.index(max(tpl))] for tpl in training_seq][i:i + window_size]) for i in range(len("".join([list(base_colors.keys())[tpl.index(max(tpl))] for tpl in training_seq])) - window_size + 1)]

        for i, base_combination in enumerate(base_combinations):
            cycle_offset = i
            spot_colors = [images[min(cycle_offset + j, num_cycles - 1)][row][col] for j in range(window_size)]

            X.append(np.array(spot_colors).flatten())
            y.append(base_combination)

    return np.array(X), np.array(y)

def base_calling_knn(images, num_cycles, num_templates, templates, num_training_templates, window_size, k):
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

    if training_files:
        print("Available training sets:")
        for i, f in enumerate(training_files):
            print(f"{i + 1}. {f}")

        choice = input("Enter the number of the training set to load, or type 'new' to create a new one: ")
        if choice.lower() == 'new':
            X_train, y_train = create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)

            # Fit the kNN classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)

            save_choice = input("Do you want to save this training set for later use? (y/n): ")
            if save_choice.lower() == 'y':
                filename = input("Enter a filename for the training set (without extension): ")
                with open(os.path.join(folder_path, f"{filename}_knn_classifier_w{window_size}_k{k}.pkl"), 'wb') as f:
                    pickle.dump(knn, f)
        else:
            file_index = int(choice) - 1
            with open(os.path.join(folder_path, training_files[file_index]), 'rb') as f:
                knn = pickle.load(f)
                match = pattern.search(training_files[file_index])
                window_size, k = int(match.group(1)), int(match.group(2))

    else:
        print("No saved training sets found.")
        X_train, y_train = create_training_set(images, num_templates, templates, num_training_templates, window_size,
                                               num_cycles, base_colors)

        # Fit the kNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        save_choice = input("Do you want to save this training set for later use? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Enter a filename for the training set (without extension): ")
            with open(os.path.join(folder_path, f"{filename}_knn_classifier_w{window_size}_k{k}.pkl"), 'wb') as f:
                pickle.dump(knn, f)

    called_bases = [''] * num_templates
    middle_index = window_size // 2
    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, math.ceil(math.sqrt(num_templates)))

        assembled_seq = []
        for cycle in range(-middle_index, num_cycles - window_size + middle_index + 1):
            spot_colors = [images[cycle + w][row][col] if 0 <= cycle + w < num_cycles else np.zeros(4) for w in range(window_size)]

            spot_colors_flat = np.array(spot_colors).reshape(1, -1)

            predicted_base_combination = knn.predict(spot_colors_flat)[0]

            assembled_seq.append(predicted_base_combination[middle_index])

        called_bases[seq_idx] = ''.join(assembled_seq)
        print("Template         ", seq_idx, ":", ''.join([base for v in templates[seq_idx] for base, color in base_colors.items() if np.array_equal(v, color)]))
        print("kNN Called Bases ", seq_idx, ":", called_bases[seq_idx])

    return called_bases