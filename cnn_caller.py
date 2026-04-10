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

    model.fit(X_train.reshape(-1, window_size, 4), onehot_encoded, epochs=iterations, batch_size=32, verbose=1)

    return model


def create_training_set_cnn(images, num_templates, templates, num_training_templates, window_size, num_cycles,
                            base_colors):
    x, y = [], []

    for training_seq_idx in range(num_training_templates):
        row, col = divmod(training_seq_idx, math.ceil(math.sqrt(num_templates)))
        training_seq = templates[training_seq_idx]

        base_combinations = [
            "".join([list(base_colors.keys())[tpl.index(max(tpl))] for tpl in training_seq][i:i + window_size]) for i in
            range(len("".join(
                [list(base_colors.keys())[tpl.index(max(tpl))] for tpl in training_seq])) - window_size + 1)]

        for i, base_combination in enumerate(base_combinations):
            cycle_offset = i
            spot_colors = [images[min(cycle_offset + j, num_cycles - 1)][row][col] for j in range(window_size)]

            x.append(np.array(spot_colors))
            y.append(base_combination)

    return np.array(x), np.array(y)


def base_calling_cnn(images, num_cycles, num_templates, templates, num_training_templates, window_size, iterations):
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

    if training_files:
        print("Available training sets:")
        for i, f in enumerate(training_files):
            print(f"{i + 1}. {f}")

        choice = input("Enter the number of the training set to load, or type 'new' to create a new one: ")
        if choice.lower() == 'new':
            x_train, y_train = create_training_set_cnn(images, num_templates, templates, num_training_templates,
                                                       window_size, num_cycles, base_colors)

            # Encode the output using one-hot encoding
            label_encoder = LabelEncoder()
            onehot_encoder = OneHotEncoder(sparse_output=False)
            integer_encoded = label_encoder.fit_transform(y_train)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

            # Fit the CNN classifier
            cnn = train_cnn_classifier(x_train, onehot_encoded, window_size, iterations)

            save_choice = input("Do you want to save this training set for later use? (y/n): ")
            if save_choice.lower() == 'y':
                filename = input("Enter a filename for the training set (without extension): ")
                cnn.save(os.path.join(folder_path, f"{filename}_cnn_classifier_w{window_size}_i{iterations}.h5"))
                with open(os.path.join(folder_path, f"{filename}_cnn_classifier_w{window_size}_i{iterations}.pkl"),
                          'wb') as f:
                    pickle.dump((label_encoder, onehot_encoder, window_size, iterations), f)

        else:
            file_index = int(choice) - 1
            training_file = training_files[file_index]
            with open(os.path.join(folder_path, f"{training_file}"), 'rb') as f:
                cnn = load_model(os.path.join(folder_path, f"{os.path.splitext(training_file)[0]}.h5"))
                label_encoder, onehot_encoder, window_size, iterations = pickle.load(f)

    else:
        print("No saved training sets found.")
        x_train, y_train = create_training_set_cnn(images, num_templates, templates, num_training_templates,
                                                   window_size, num_cycles, base_colors)

        # Encode the output using one-hot encoding
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse_output=False)
        integer_encoded = label_encoder.fit_transform(y_train)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        # Fit the CNN classifier
        cnn = train_cnn_classifier(x_train, onehot_encoded, window_size, iterations)

        save_choice = input("Do you want to save this training set for later use? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Enter a filename for the training set (without extension): ")
            with open(os.path.join(folder_path, f"{filename}_cnn_classifier_w{window_size}_i{iterations}.pkl"),
                      'wb') as f:
                pickle.dump((cnn, label_encoder, onehot_encoder, window_size, iterations), f)

    called_bases = [''] * num_templates
    middle_index = window_size // 2
    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, math.ceil(math.sqrt(num_templates)))

        assembled_seq = []
        for cycle in range(-middle_index, num_cycles - window_size + middle_index + 1):
            spot_colors = [images[cycle + w][row][col] if 0 <= cycle + w < num_cycles else np.zeros(4) for w in
                           range(window_size)]
            spot_colors_flat = np.array(spot_colors).reshape((1, window_size, 4))

            predicted_base_combination_idx = np.argmax(cnn.predict(spot_colors_flat), axis=1)[0]
            assembled_seq.append(label_encoder.inverse_transform([predicted_base_combination_idx])[0][middle_index])

        called_bases[seq_idx] = ''.join(assembled_seq)
        print("Template         ", seq_idx, ":", ''.join(
            [base for v in templates[seq_idx] for base, color in base_colors.items() if np.array_equal(v, color)]))
        print("cnn Called Bases ", seq_idx, ":", called_bases[seq_idx])

    return called_bases
