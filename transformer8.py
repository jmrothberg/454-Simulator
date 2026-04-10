#Transformer_base_calling images to letter bases for 454.Bio.  JMR April 5 2023

import math
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Get the current working directory
cwd = os.getcwd()
print ("current working directory")

# Define the folder name
folder_name = 'Sim Models'

# Create the folder path
folder_path = os.path.join(cwd, folder_name)

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print ("Made New Folder ", folder_path)
def onehot_to_base(predicted_colors):
    base_colors = {
        'A': (1.0, 0.0, 0.0, 0.0),
        'C': (0.0, 1.0, 0.0, 0.0),
        'G': (0.0, 0.0, 1.0, 0.0),
        'T': (0.0, 0.0, 0.0, 1.0)
    }

    predicted_base_seq = []
    for i in range(0, len(predicted_colors), 4):
        color_values = predicted_colors[i:i+4]
        predicted_base = max(base_colors, key=lambda k: np.dot(color_values, base_colors[k]))
        predicted_base_seq.append(predicted_base)

    return predicted_base_seq

def create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors):
    print ("create_training_set")
    X_train, y_train = [], []
    for training_seq_idx in range(num_training_templates):

        row, col = divmod(training_seq_idx, math.ceil(math.sqrt(num_templates)))
        training_seq = templates[training_seq_idx]

        base_combinations = ["".join([list(base_colors.keys())[tpl.index(max(tpl))] for tpl in training_seq][i:i + window_size]) for i in range(len("".join([list(base_colors.keys())[tpl.index(max(tpl))] for tpl in training_seq])) - window_size + 1)]

        for i, base_combination in enumerate(base_combinations):
            cycle_offset = i
            spot_colors = [images[min(cycle_offset + j, num_cycles - 1)][row][col] for j in range(window_size)]

            expected_colors = [np.array(base_colors[base]) for base in base_combination]

            X_train.append(np.array(spot_colors))
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

def transformer_base_calling(images, num_cycles, num_templates, templates, num_training_templates, window_size):
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
    if model_files:
        print("Available models:")
        for i, f in enumerate(model_files):
            print(f"{i + 1}. {f}")

        choice = input("Enter the number of the model to load, or name (must have letters/without extension) to create a new one: ")
        if not choice.isdigit():
            save_choice = input("Do you want to save this model (hit return to skip): ")
            if save_choice:
                filename = choice
            # Create a new model and train it
            X_train, y_train = create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)
            input_layer = Input(shape=(window_size, 4))
            x = MultiHeadAttention(num_heads=8, key_dim=4)(input_layer, input_layer)
            x = LayerNormalization(epsilon=1e-6)(Add()([input_layer, x]))
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.2)(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.2)(x)
            #x = Flatten()(x)
            #output_layer = Dense(4 * window_size, activation='softmax')(x)
            x = Flatten()(x)
            output_layer = Dense(4 * window_size)(x)
            output_layer = tf.reshape(output_layer, (-1, window_size, 4))
            output_layer = tf.nn.softmax(output_layer, axis=-1)
            output_layer = Flatten()(output_layer)

            transformer_model = Model(inputs=[input_layer], outputs=[output_layer])
            #transformer_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            transformer_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')

            transformer_model.fit(X_train, y_train, batch_size=32, epochs=10)

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
        # Create a new model and train it
        X_train, y_train = create_training_set(images, num_templates, templates, num_training_templates, window_size, num_cycles, base_colors)
        input_layer = Input(shape=(window_size, 4))
        x = MultiHeadAttention(num_heads=8, key_dim=4)(input_layer, input_layer)
        x = LayerNormalization(epsilon=1e-6)(Add()([input_layer, x]))
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        #x = Flatten()(x)
        #output_layer = Dense(4 * window_size, activation='softmax')(x)
        x = Flatten()(x)
        output_layer = Dense(4 * window_size)(x)
        output_layer = tf.reshape(output_layer, (-1, window_size, 4))
        output_layer = tf.nn.softmax(output_layer, axis=-1)
        output_layer = Flatten()(output_layer)

        transformer_model = Model(inputs=[input_layer], outputs=[output_layer])
        #transformer_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        transformer_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')

        transformer_model.fit(X_train, y_train, batch_size=32, epochs=10)

        if save_choice:
            transformer_model.save(os.path.join(folder_path, f"{filename}_transformer_model_w{window_size}.h5"))
    # reset window size if saved model not the user input
    called_bases = [''] * num_templates
    middle_index = window_size // 2
    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, math.ceil(math.sqrt(num_templates)))
        assembled_seq = []

        for cycle in range(-middle_index, num_cycles - window_size + middle_index + 1):
            spot_colors = [images[cycle + w][row][col] if 0 <= cycle + w < num_cycles else np.zeros(4) for w in range(window_size)]
            spot_colors = np.array(spot_colors) / 255.0
            spot_colors = spot_colors.reshape(1, window_size, 4)
            predicted_colors = transformer_model.predict(spot_colors)[0]

            predicted_base_seq = onehot_to_base(predicted_colors)
            assembled_seq.append(predicted_base_seq[middle_index])

        called_bases[seq_idx] = ''.join(assembled_seq)
        print("Template         ", seq_idx, ":", ''.join([base for v in templates[seq_idx] for base, color in base_colors.items() if np.array_equal([val / 255.0 for val in v], color)]))
        print("Transformer Called bases ", seq_idx, ":", called_bases[seq_idx])

    return called_bases
