

# Multi-Sim 2 Multi-Caller strand-simulator, strand visualization, image based simulator, images, sequence plots, histogram, select your basecallers.
# Based on Sequencing by Synthesis (Invented by Jonathan Rothberg) with 454 E-wave technology and Lightning terminators.
# Jonathan Rothberg, 454 Bio March & April, 13 2023
import os
import sys
import math
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Base callers (imports are silent — run MultiSim2.py to drive the pipeline)
from knn_caller4 import base_calling_knn
from transformer8 import transformer_base_calling
from causal_transformer import causal_transformer_base_calling
from multipass3 import base_calling_multipass
from lagleaddeath import estimate_lag_lead_percentages
from cnn_caller import base_calling_cnn

def get_rgb_from_four_color_channels(a, c, g, t):
    r = t + a
    green = c + a
    b = g

    total = r + green + b

    # Clip the values to the maximum 255 - if algorithms working this should not be needed
    r = int(min(r * bright_scale, 255))
    green = int(min(green * bright_scale, 255))
    b = int(min(b * bright_scale, 255))

    return r, green, b

def convert_vectors_to_bases(
        vectors):  # This is the color of the dyes on the bases added, you use to convert to the base. This can be simpler, and more flexible.
    bases = []
    for v in vectors:
        if np.array_equal(v, [255, 0, 0, 0]):
            bases.append('A')
        elif np.array_equal(v, [0, 255, 0, 0]):
            bases.append('C')
        elif np.array_equal(v, [0, 0, 255, 0]):
            bases.append('G')
        elif np.array_equal(v, [0, 0, 0, 255]):
            bases.append('T')
    return ''.join(bases)

def convert_bases_to_vectors(letter_templates):
    base_colors = {
        'A': [255.0, 0.0, 0.0, 0.0],
        'C': [0.0, 255.0, 0.0, 0.0],
        'G': [0.0, 0.0, 255.0, 0.0],
        'T': [0.0, 0.0, 0.0, 255.0]
    }

    vector_templates = []

    for template in letter_templates:
        vector_template = [base_colors[base] for base in template]
        vector_templates.append(vector_template)

    return vector_templates

'''
def convert_vectors_to_bases(vectors):
    base_colors = {(255, 0, 0, 0): 'A', (0, 255, 0, 0): 'C', (0, 0, 255, 0): 'G', (0, 0, 0, 255): 'T'}
    return ''.join(map(lambda v: base_colors[tuple(v)], vectors))
'''
def generate_random_templates(number_of_templates, template_length, key):  # Letter templates used by strand simulator
    print("generate_random_templates")
    bases = ["A", "C", "G", "T"]

    random_templates = []

    for _ in range(number_of_templates):
        random_template = [random.choice(list(dye_dict.keys())) for _ in range(template_length)]
        random_template = key + "".join(random_template)
        random_templates.append(random_template)
    return random_templates

# Simulate sequencing usings strands. Uses base letters not vectors
def simulate_cycle(strands, template, cycle):
    # print ("simulate_cycle")
    # Loop over the strands
    for i, strand in enumerate(strands):
        # Skip the iteration if the strand is already longer than the template
        if len([x for x in strand if (x != ' ' and x != '.')]) >= len(template):
            continue
        # Check if the strand unblocks during the UV window and extends and then unblocks again
        # Generate the random number for the first unblocking event
        uv_extend_count = 0
        rand_num1 = np.random.random()
        if strand[-1] in ['yellow', 'green', 'blue', 'red'] and rand_num1 < (1 - np.exp(-uv_time / tauUV)):
            # Calculate the time elapsed in the UV window for the first event
            time_elapsed1 = -tauUV * np.log(1 - rand_num1)
            # Subtract the time elapsed from the original uv_time
            nuv_time = uv_time - time_elapsed1
            # Remove the dye label
            strand = strand[:-1]
            uv_unblock_count = 1
            # Loop to add bases during the UV window which if cleaved after addition are invisible they move you forward in a frame!
            while nuv_time > 0:
                rand_num2 = np.random.random()
                if strand[-1] not in ['yellow', 'green', 'blue', 'red'] and rand_num2 < (1 - np.exp(-nuv_time / tauEX)):
                    uv_extend_count = uv_extend_count + 1
                    time_elapsed2 = -tauEX * np.log(1 - rand_num2)
                    nuv_time = nuv_time - time_elapsed2
                    next_base_in_template = template[len([x for x in strand if (x != ' ' and x != '.')])]
                    complement_base = complement_dict[next_base_in_template]
                    dye = dye_dict[complement_base]
                    if strand[-1].islower() and strand[-2].islower():
                        strand += [complement_base, dye]
                    elif strand[-1].islower():
                        strand += [' ', complement_base, dye]
                    else:
                        strand += [' ', ' ', complement_base, dye]
                    rand_num3 = np.random.random()
                    if rand_num3 < (1 - np.exp(-nuv_time / tauUV)):
                        uv_unblock_count = uv_unblock_count + 1
                        time_elapsed3 = -tauUV * np.log(1 - rand_num3)
                        nuv_time = nuv_time - time_elapsed3
                        last_element = strand[-2].lower()
                        # print("uv_unblock_count, Removed: ", uv_unblock_count, strand[-(6-uv_unblock_count):])
                        if uv_unblock_count > 4:
                            uv_unblock_count = 4
                        strand[-(6 - uv_unblock_count):] = [last_element]
                        # print("Replaced with: ", [last_element])
                else:
                    # Break out of the loop if the if statement is not true
                    break

        extension_count = uv_extend_count

        # UV Killing of strand. Updated to be function of uv_time
        if strand[-1] != 'dead' and np.random.random() < p_die * uv_time:
            strand.append('dead')

        # Check if the strand extends during the second part of the cycle
        dark_count = 0
        rand_num1 = np.random.random()
        # Do you extend in the ex_time window
        if strand[-1] not in ['dead', 'yellow', 'green', 'blue', 'red'] and rand_num1 < (1 - np.exp(-ex_time / tauEX)):
            time_elapsed1 = -tauEX * np.log(1 - rand_num1)
            nex_time = ex_time - time_elapsed1
            percent_blocked = (1 - p_dark * uv_time) ** cycle  # amount of labeled and blocked nucleotides
            rand_numDark = np.random.random()
            if len([x for x in strand if (x != ' ' and x != '.')]) >= len(template):
                break
            # Extend the strand and add a new dye - This is NORMAL behavior,
            if rand_numDark < percent_blocked:  # Do you put on a good base with dye e.g. blocked.
                next_base_in_template = template[len([x for x in strand if (x != ' ' and x != '.')])]
                complement_base = complement_dict[next_base_in_template]
                dye = dye_dict[complement_base]
                if strand[-1].islower() and strand[-2].islower():
                    strand += [complement_base, dye]
                elif strand[-1].islower():
                    strand += [' ', complement_base, dye]
                else:
                    strand += [' ', ' ', complement_base, dye]
                extension_count += 1
            else:
                # Dark base loop. N in the print for first dark base so we know keeps frame.  Assume unblocked and dark.
                # Need to check time remaining before each extension
                # print ("Added first dark base, cycle, percent unblocked, nex_time :", cycle , percent_unblocked, nex_time)
                next_base_in_template = template[len([x for x in strand if (x != ' ' and x != '.')])]
                complement_base = 'N'  # it is dark so you don't know what is added :)
                # Spacing so if lower case based before you need to move over less ot make the base align with the other bases in print outs
                # all 3 of these conditions occur here and in two places above
                if strand[-1].islower() and strand[-2].islower():
                    strand += [complement_base]
                elif strand[-1].islower():
                    strand += [' ', complement_base]
                else:
                    strand += [' ', ' ', complement_base]
                dark_count = dark_count + 1
                while nex_time > 0:
                    if len([x for x in strand if (x != ' ' and x != '.')]) >= len(template):
                        break
                    rand_num2 = np.random.random()
                    # Use remaining time nex_time to adjust probabilities
                    if strand[-1] not in ['dead', 'yellow', 'green', 'blue', 'red'] and rand_num2 < (
                            1 - np.exp(-nex_time / tauEX)):
                        time_elapsed2 = -tauEX * np.log(1 - rand_num2)  # how long did it take
                        nex_time = nex_time - time_elapsed2  # lose the time in the extension phase
                        rand_numDark = np.random.random()
                        # if you extend do you get a dark base?
                        if rand_numDark > percent_blocked:
                            # print ("Added addional out of phase dark bases in cycle, percent unblocked, nex_time :", cycle , percent_unblocked, nex_time)
                            next_base_in_template = template[len([x for x in strand if (x != ' ' and x != '.')])]
                            complement_base = 'n'  # it is dark so you don't know what is added :)
                            strand += [complement_base]
                            dark_count = dark_count + 1
                        else:
                            break
                    else:
                        # Break out of the loop if the if statement is not true
                        break

        # This makes a strand LAG because no bases where added in cycle
        # Insert the spaces before the dye which stays at end
        if strand[-1] != 'dead' and extension_count == 0 and dark_count == 0:
            if strand[-1] in ['yellow', 'green', 'blue', 'red']:
                strand.insert(-1, " ")
                strand.insert(-1, " ")
                strand.insert(-1, ".")
            else:
                strand.append(" ")
                strand.append(" ")
                strand.append(".")

        # Update the global strands list with the updated local strand from the for loop
        strands[i] = strand
    return

def simulate_sequencing(letter_templates, num_templates_to_process, num_cycles, num_strands):
    print("simulate_sequencing")
    images = []
    allstrands = []

    # Initialize the strands
    for i in range(num_templates_to_process):
        seq = letter_templates[i]

        strands = []
        for i2 in range(num_strands):
            strands.append([' ', ' '])

        # Initialize the dye counts
        dye_counts = np.zeros((num_cycles, 4))

        # Loop over the cycles and simulate sequencing
        for cycle in range(num_cycles):
            simulate_cycle(strands, seq, cycle)
            # Terminal dye per strand (matches 454Sim13): include dye-before-'dead' for early deaths.
            terminal_dye = {'yellow': 0, 'green': 1, 'blue': 2, 'red': 3}
            for strand in strands:
                if not strand:
                    continue
                last = strand[-1]
                if last in terminal_dye:
                    dye_counts[cycle][terminal_dye[last]] += 1
                elif last == 'dead' and len(strand) >= 2 and strand[-2] in terminal_dye:
                    dye_counts[cycle][terminal_dye[strand[-2]]] += 1

        allstrands.append(strands)
        images.append(dye_counts)

    return images, allstrands

def generate_images_with_noise(templates, number_of_templates, num_cycles, lag_percentage, lead_percentage,
                               noise_percentage, death_percentage):
    image_dim = int(np.ceil(np.sqrt(number_of_templates)))

    images = []
    for cycle in range(num_cycles):
        image = np.zeros((image_dim, image_dim, 4), dtype=float)

        for i, seq in enumerate(templates):
            row, col = divmod(i, image_dim)
            current_color = np.array(seq[cycle], dtype=float)

            # Calculate the alive factor based on the death_percentage for the current cycle
            alive_factor = (1 - death_percentage) ** cycle

            if cycle > 0:
                cumulative_lag_color = np.zeros(4)
                for prev_cycle in range(cycle):
                    lagging_color = np.array(templates[i][prev_cycle], dtype=float)
                    lag_multiplier = (1 - lag_percentage) ** (cycle - prev_cycle) * lag_percentage

                    lag_noise_range = lag_multiplier * noise_percentage
                    lag_noise = random.uniform(-lag_noise_range, lag_noise_range)
                    noisy_lag_percentage = lag_multiplier + lag_noise

                    cumulative_lag_color += lagging_color * noisy_lag_percentage

                # Subtract the lagging color value from the current color and add the lagging color vector
                current_color[np.argmax(current_color)] -= np.sum(cumulative_lag_color)
                current_color += cumulative_lag_color

            if cycle < num_cycles - 1:
                cumulative_lead_color = np.zeros(4)
                for lead_cycle in range(cycle + 1, min(len(templates[i]), cycle + 1 + cycle)):
                    leading_color = np.array(templates[i][lead_cycle], dtype=float)
                    lead_multiplier = (1 - lead_percentage) ** (lead_cycle - cycle) * lead_percentage

                    lead_noise_range = lead_multiplier * noise_percentage
                    lead_noise = random.uniform(-lead_noise_range, lead_noise_range)
                    noisy_lead_percentage = lead_multiplier + lead_noise

                    cumulative_lead_color += leading_color * noisy_lead_percentage

                # Subtract the leading color value from the current color and add the leading color vector
                current_color[np.argmax(current_color)] -= np.sum(cumulative_lead_color)
                current_color += cumulative_lead_color

            current_color *= alive_factor
            image[row][col] = current_color

        images.append(image)

    return images

# This simple base-calling working with vector data for each base in template and in the image file which is slices of the templates for each cycle.
def base_calling_single_image(images, num_cycles, num_templates):
    print("base_calling_single_image")
    base_colors = {
        'A': (255.0, 0.0, 0.0, 0.0),
        'C': (0.0, 255.0, 0.0, 0.0),
        'G': (0.0, 0.0, 255.0, 0.0),
        'T': (0.0, 0.0, 0.0, 255.0)
    }
    # L2 unit direction: scale-invariant vs overall brightness; ties → N (not first channel).
    min_l2_signal = 1e-9
    min_l2_component = 0.45  # min dominant axis on unit vector for a confident call
    margin_vs_runner_up = 1.18
    tie_band = 0.03  # top-two L2 components within this → ambiguous

    called_bases = [''] * num_templates
    for seq_idx in range(num_templates):
        for cycle in range(num_cycles):
            row, col = divmod(seq_idx, math.ceil(math.sqrt(num_templates)))
            spot = np.asarray(images[cycle][row][col], dtype=np.float64).ravel()[:4]
            l2 = float(np.linalg.norm(spot))
            if l2 < min_l2_signal:
                called_bases[seq_idx] += 'N'
                continue
            u = spot / l2
            order = np.sort(u)
            low, high = float(order[-2]), float(order[-1])
            max_index = int(np.argmax(u))
            if high - low < tie_band:
                called_bases[seq_idx] += 'N'
            elif high < min_l2_component or (low > 1e-12 and high < margin_vs_runner_up * low):
                called_bases[seq_idx] += 'N'
            else:
                called_bases[seq_idx] += ['A', 'C', 'G', 'T'][max_index]

        print(f"Single Image {seq_idx + 1}: {called_bases[seq_idx]}")

    return called_bases

def display_images(images):
    print("display_images")
    num_images = len(images)
    num_rows = int(num_images ** 0.5)
    num_cols = num_images // num_rows + (1 if num_images % num_rows > 0 else 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

    for i, img in enumerate(images):
        row, col = divmod(i, num_cols)
        ax = axes[row][col] if num_rows > 1 else axes[col]

        rgb_image = np.array([[get_rgb_from_four_color_channels(*spot_color) for spot_color in row] for row in img])

        ax.imshow(rgb_image)
        ax.set_title(f"Image {i + 1}")

    plt.tight_layout()
    # Save to a temporary file
    temp_file = filename + '_array.png'
    print("Saving Array: ", temp_file)
    plt.savefig(os.path.join(folder_path, temp_file), bbox_inches='tight')
    plt.show()

def plot_single_cycle_images(images, num_templates_to_process, num_cycles):
    print("plot_single_cycle_images")
    num_rows = int(np.ceil(np.sqrt(num_templates_to_process)))
    num_cols = int(np.ceil(num_templates_to_process / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

    for i in range(num_templates_to_process):
        row, col = divmod(i, num_cols)
        ax = axes[row][col] if num_rows > 1 else axes[col]

        data = np.zeros((4, num_cycles))
        for cycle in range(num_cycles):
            data[:, cycle] = images[cycle][row][col][:4]

        x = np.arange(num_cycles)

        colors = ['yellow', 'green', 'blue', 'red']

        bottom = np.zeros(num_cycles)
        for j, color in enumerate(colors):
            ax.bar(x, data[j], color=color, bottom=bottom)
            bottom += data[j]

        ax.set_title(f"Template {i + 1}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Vector Value")

    plt.tight_layout()
    temp_file = filename + '_seq.png'
    print("Saving Seq: ", temp_file)
    plt.savefig(os.path.join(folder_path, temp_file), bbox_inches='tight')
    plt.show()

def plot_histograms(allstrands, num_templates_to_process, num_cycles):
    print("plot_histograms")
    num_rows = int(np.ceil(np.sqrt(num_templates_to_process)))
    num_cols = int(np.ceil(num_templates_to_process / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 2.5),
                             gridspec_kw={'wspace': 0.4, 'hspace': 0.6})

    for i, strands in enumerate(allstrands):
        row, col = divmod(i, num_cols)
        ax = axes[row][col] if num_rows > 1 else axes[col]
        strand_lengths = [len([x for x in strand if x != ' ' and x != '.' and len(x) == 1]) for strand in strands]
        # Perfect reads in phase, which means need to terminate if you hit a . or n or N
        strand_lengths_in_phase = []
        for strand in strands:
            length = 0
            for character in strand:
                if character not in ['.', 'n',
                                     ' '] and character.isupper():  # I removed 'N' so you are allowed an in frame N
                    length += 1  # Can make logic more complex if you want to allow a .n to be in frame
                elif character == ' ':
                    pass
                else:
                    break
            strand_lengths_in_phase.append(length)

        phase_counts = dict(Counter(strand_lengths_in_phase))
        xp = list(phase_counts.keys())
        yp = list(phase_counts.values())

        length_counts = dict(Counter(strand_lengths))
        x = list(length_counts.keys())
        y = list(length_counts.values())

        ax.bar(x, y, color='blue', alpha=0.5)
        ax.bar(xp, yp, color='red', alpha=0.5)
        plt.xlabel("Strand Length")
        plt.ylabel("Number of Strands")
        ax.set_title(f"Template {i + 1}")
    legend_elements = [Line2D([0], [0], color='blue', lw=4, label='All Strands'),
                       Line2D([0], [0], color='red', lw=4, label='In Phase')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=2)
    temp_file = filename + '_dist.png'
    print ("Saving Histogram file: ", temp_file)
    plt.savefig(os.path.join(folder_path, temp_file), bbox_inches='tight')
    plt.show()

def get_input(prompt, default_value):
    user_input = input(f"{prompt} (default: {default_value}): ").strip()
    return default_value if user_input == "" else float(user_input)

def get_input_string(prompt, default_value):
    user_input = input(f"{prompt} (default: {default_value}): ").strip()
    return default_value if user_input == "" else user_input

if __name__ == "__main__":
    print("main")
    dye_dict = {'A': 'yellow', 'C': 'green', 'G': 'blue', 'T': 'red'}
    complement_dict = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'N': 'N',
                       '-': '-'}  # don't use complement to make compatible with array methods

    number_of_templates = int(get_input("Number of templates to generate: ", 200))
    num_training_templates = int(get_input("Number of training templates: ", 160))
    template_length = int(get_input("Template length: ", 100))
    key = get_input_string(
        "Enter a key sequence", "ACGTTGCA")

    num_cycles = int(get_input("Number of cycles (< Template Length): ", 60))
    window = int(get_input("Window size for analysis: ", 5))
    num_passes: int = int(get_input("num_passes for multipass_accuracy: ", 3))
    k = int(get_input("k value for kNN: ", 5))

    template = []
    keystart = 0

    letter_templates = generate_random_templates(number_of_templates, template_length, key)
    vector_templates = convert_bases_to_vectors(letter_templates)

    image_dim = int(np.ceil(np.sqrt(number_of_templates)))
    images = []

    # Get the input variables for each method
    method_choice = input("Choose method: (1) strand-based simulator, (2) simple lead lag noise simulator: ").strip()
    if not method_choice:
        method_choice = "1"
    if method_choice == "1":
        tauUV = get_input("Enter tauUV : ", .3)
        uv_time = get_input("Enter uv_time : ", 1)
        tauEX = get_input("Enter tauEX : ", 10)
        ex_time = get_input("Enter ex_time : ", 100)
        p_die = get_input("Enter p_die percent strands die per second of UV per cycle: ", .01)
        p_dark = get_input("Enter p_dark percent dark bases per second of UV per cycle: ", .005)
        num_strands = int(get_input("Number of strands: ", 255))
        filename = f"{number_of_templates}x{num_cycles}_tauUV_{tauUV}_uv_time_{uv_time}_tauEX_{tauEX}_ex_time_{ex_time}_p_die_{p_die}_p_dark_{p_dark}_strands_{num_strands}.txt"

        bright_scale = 255 / num_strands  # so it it is compatible with array based simulator  # Matched to image based simulator with 255 as maximum
        allstrands = []
        # Use simulate_sequences
        images, allstrands = simulate_sequencing(letter_templates, number_of_templates, num_cycles, num_strands)

        images_array = np.array(images)
        # Calculate the required padding
        padding = image_dim * image_dim - number_of_templates

        # Pad the images array with zeros
        if padding > 0:
            images_array = np.vstack((images_array, np.zeros((padding, num_cycles, 4))))

        # Reshape images_array to have dimensions (number_of_cycles, image_dim, image_dim, 4)
        reshaped_images = []
        for cycle in range(num_cycles):
            cycle_image = images_array[:, cycle, :].reshape((image_dim, image_dim, 4))
            reshaped_images.append(cycle_image)
        images = reshaped_images  # to be compatible with code below

    elif method_choice == "2":
        # Use generate_images_with_noise and ask for lag, lead, noise, death percentage with default of 1%
        lag_percentage = get_input("Enter lag percentage", .01)
        lead_percentage = get_input("Enter lead percentage", .01)
        noise_percentage = get_input("Enter noise percentage", .05)
        death_percentage = get_input("Enter death percentage", .005)

        filename = f"{number_of_templates}x{num_cycles}_lag_{lag_percentage}_lead_{lead_percentage}_noise_{noise_percentage}_death_{death_percentage}.txt"
        bright_scale = 1  # since already 255 scale for image method
        images = generate_images_with_noise(vector_templates, number_of_templates, num_cycles, lag_percentage,
                                            lead_percentage, noise_percentage, death_percentage)
        reshaped_images = images  # No need to reshape for generate_images_with_noise
    else:
        print("Invalid choice, exiting.")
        sys.exit(1)

    called_sequences_single_image = []
    called_sequences_integrated = []
    called_sequences_multipass = []
    called_sequences_knn = []
    called_sequences_bidir = []
    called_sequences_causal = []

    single_image_accuracy_sum = 0
    integrated_accuracy_sum = 0
    knn_accuracy_sum = 0
    transformer_accuracy_sum = 0

    # Define method names
    methods = [
        "single_image",
        "multipass",
        "kNN",
        "cnn",
        "bidir encoder",
        "causal transformer",
        "estimate lag, led, noise, death"
    ]

    # Let the user select methods (press Enter or type 'all' to run every real method)
    print("Available methods:")
    for idx, method_name in enumerate(methods, start=1):
        print(f"{idx}. {method_name}")
    selected_methods_input = input("Enter indices separated by space, or 'all' / Enter to run all real methods: ").strip()
    if selected_methods_input == "" or selected_methods_input.lower() == "all":
        # All real methods: 1-6 + 7 (estimate lag/lead/death)
        selected_methods_indices = [1, 2, 3, 4, 5, 6, 7]
        auto_train = True   # skip interactive load/save prompts for ML callers
    else:
        selected_methods_indices = list(map(int, selected_methods_input.split()))
        auto_train = False
    selected_methods = {methods[idx - 1]: False for idx in range(1, len(methods) + 1)}

    for idx in selected_methods_indices:
        selected_methods[methods[idx - 1]] = True

    if selected_methods["estimate lag, led, noise, death"]:
        best_lag, best_lead, best_death, noise_lag, noise_lead, noise_death, min_error = estimate_lag_lead_percentages(images, key, keystart,
                                                                                                number_of_templates)
        print(f"Estimated Lag Percentage   : {best_lag * 100:.2f}%")
        print(f"Estimated Lead Percentage  : {best_lead * 100:.2f}%")
        print(f"Estimated Death Percentage : {best_death * 100:.2f}%")

        # lagleaddeath: dimensionless sensitivity (model shift / data residual) per pathway, not a true %
        print(f"Lag-path sensitivity index   : {noise_lag:.4f}")
        print(f"Lead-path sensitivity index  : {noise_lead:.4f}")
        print(f"Death-path sensitivity index : {noise_death:.4f}")

        print("Estimated min_error (SSE)    :", min_error)

    # Call the selected methods
    if selected_methods["single_image"]:
        single_image_accuracy_sum = 0
        called_sequences_single_image = base_calling_single_image(images, num_cycles - window // 2,
                                                                  number_of_templates)

    if selected_methods["multipass"]:
        # window = int(get_input("Multipass window size: ", 5))
        # num_passes = int(get_input("num_passes for multipass_accuracy: ", 3))
        multipass_accuracy_sums = [0] * num_passes
        called_sequences_multipass = base_calling_multipass(images, num_cycles, key, number_of_templates, window,
                                                            num_passes)

    if selected_methods["kNN"]:
        knn_accuracy_sum_train = 0
        knn_accuracy_sum_test = 0

        called_sequences_knn = base_calling_knn(images, num_cycles, number_of_templates, vector_templates,
                                                num_training_templates, window, k, auto_train=auto_train)

    if selected_methods["bidir encoder"]:
        bidir_accuracy_sum_train = 0
        bidir_accuracy_sum_test = 0
        called_sequences_bidir = transformer_base_calling(images, num_cycles, number_of_templates,
                                                          vector_templates, num_training_templates, window, auto_train=auto_train)

    if selected_methods["causal transformer"]:
        causal_accuracy_sum_train = 0
        causal_accuracy_sum_test = 0
        called_sequences_causal = causal_transformer_base_calling(images, num_cycles, number_of_templates,
                                                                  vector_templates, num_training_templates, window, auto_train=auto_train)

    if selected_methods["cnn"]:
            cnn_accuracy_sum_train = 0
            cnn_accuracy_sum_test = 0
            called_sequences_integrated = base_calling_cnn(images, num_cycles, number_of_templates, vector_templates, num_training_templates, window, num_passes, auto_train=auto_train)

    num_test_templates = number_of_templates - num_training_templates

    for i in range(number_of_templates):
        original_seq = vector_templates[i]
        original_seq_bases = letter_templates[i]
        is_test = (i >= num_training_templates)
        split_tag = "TEST" if is_test else "train"
        print(f"Original template {i + 1} [{split_tag}]       : {original_seq_bases}")

        if selected_methods.get("single_image"):
            single_image_called_seq = called_sequences_single_image[i]
            n_cmp = min(len(original_seq_bases), len(single_image_called_seq))
            single_image_accuracy = sum(1 for t, c in zip(original_seq_bases, single_image_called_seq) if t == c) / max(n_cmp, 1) * 100
            single_image_accuracy_sum += single_image_accuracy
            print(
                f"Single Image sequence {i + 1}        : {single_image_called_seq} (Accuracy: {single_image_accuracy:.2f}%)")

        if selected_methods.get("multipass"):
            multipass_seqs = called_sequences_multipass[i]
            for p, multipass_seq in enumerate(multipass_seqs):
                n_cmp = min(len(original_seq_bases), len(multipass_seq))
                multipass_accuracy = sum(1 for t, c in zip(original_seq_bases, multipass_seq) if t == c) / max(n_cmp, 1) * 100
                multipass_accuracy_sums[p] += multipass_accuracy
                print(
                    f"Multi-Pass sequence {i + 1} (Pass {p + 1}) : {multipass_seq} (Accuracy: {multipass_accuracy:.2f}%)")

        if selected_methods.get("kNN"):
            knn_seq = called_sequences_knn[i]
            n_cmp = min(len(original_seq_bases), len(knn_seq))
            knn_accuracy = sum(1 for t, c in zip(original_seq_bases, knn_seq) if t == c) / max(n_cmp, 1) * 100
            if is_test:
                knn_accuracy_sum_test += knn_accuracy
            else:
                knn_accuracy_sum_train += knn_accuracy
            print(f"kNN sequence {i + 1}                 : {knn_seq} (Accuracy: {knn_accuracy:.2f}%)")

        if selected_methods.get("bidir encoder"):
            bidir_seq = called_sequences_bidir[i]
            n_cmp = min(len(original_seq_bases), len(bidir_seq))
            bidir_accuracy = sum(1 for t, c in zip(original_seq_bases, bidir_seq) if t == c) / max(n_cmp, 1) * 100
            if is_test:
                bidir_accuracy_sum_test += bidir_accuracy
            else:
                bidir_accuracy_sum_train += bidir_accuracy
            print(
                f"Bidir Encoder sequence {i + 1}      : {bidir_seq} (Accuracy: {bidir_accuracy:.2f}%)")

        if selected_methods.get("causal transformer"):
            causal_seq = called_sequences_causal[i]
            n_cmp = min(len(original_seq_bases), len(causal_seq))
            causal_accuracy = sum(1 for t, c in zip(original_seq_bases, causal_seq) if t == c) / max(n_cmp, 1) * 100
            if is_test:
                causal_accuracy_sum_test += causal_accuracy
            else:
                causal_accuracy_sum_train += causal_accuracy
            print(
                f"Causal Transformer seq {i + 1}      : {causal_seq} (Accuracy: {causal_accuracy:.2f}%)")

        if selected_methods.get("cnn"):
            integrated_seq = called_sequences_integrated[i]
            n_cmp = min(len(original_seq_bases), len(integrated_seq))
            integrated_accuracy = sum(1 for t, c in zip(original_seq_bases, integrated_seq) if t == c) / max(n_cmp, 1) * 100
            if is_test:
                cnn_accuracy_sum_test += integrated_accuracy
            else:
                cnn_accuracy_sum_train += integrated_accuracy
            print(f"cnn Called sequence {i + 1}          : {integrated_seq} (Accuracy: {integrated_accuracy:.2f}%)")

    # Calculate the overall accuracy for each method
    # Physics methods: all templates are unseen (no training data used)
    # ML methods: report TEST-only accuracy (templates the model never trained on)
    print(f"\n{'='*70}")
    print(f"ACCURACY SUMMARY  (trained on {num_training_templates}, tested on {num_test_templates} unseen templates)")
    print(f"{'='*70}")

    if selected_methods.get("single_image"):
        accuracy_single_image = single_image_accuracy_sum / number_of_templates
        print(f"Single Image (all {number_of_templates} templates, no training): {accuracy_single_image:.2f}%")

    if selected_methods.get("multipass"):
        accuracy_multipass = [sum_acc / number_of_templates for sum_acc in multipass_accuracy_sums]
        for p, acc in enumerate(accuracy_multipass):
            print(f"Multi-Pass   (all {number_of_templates} templates, no training) Pass {p + 1}: {acc:.2f}%")
            multilastpassaccuracy = acc

    if selected_methods.get("kNN"):
        acc_train = knn_accuracy_sum_train / max(num_training_templates, 1)
        acc_test  = knn_accuracy_sum_test  / max(num_test_templates, 1)
        print(f"kNN          TEST ({num_test_templates} unseen): {acc_test:.2f}%   (train {num_training_templates}: {acc_train:.2f}%)")

    if selected_methods.get("bidir encoder"):
        acc_train = bidir_accuracy_sum_train / max(num_training_templates, 1)
        acc_test  = bidir_accuracy_sum_test  / max(num_test_templates, 1)
        print(f"Bidir Enc    TEST ({num_test_templates} unseen): {acc_test:.2f}%   (train {num_training_templates}: {acc_train:.2f}%)")

    if selected_methods.get("causal transformer"):
        acc_train = causal_accuracy_sum_train / max(num_training_templates, 1)
        acc_test  = causal_accuracy_sum_test  / max(num_test_templates, 1)
        print(f"Causal Trans TEST ({num_test_templates} unseen): {acc_test:.2f}%   (train {num_training_templates}: {acc_train:.2f}%)")

    if selected_methods.get("cnn"):
        acc_train = cnn_accuracy_sum_train / max(num_training_templates, 1)
        acc_test  = cnn_accuracy_sum_test  / max(num_test_templates, 1)
        print(f"CNN          TEST ({num_test_templates} unseen): {acc_test:.2f}%   (train {num_training_templates}: {acc_train:.2f}%)")

    print(f"{'='*70}")

    #figures = input("Y or any key to display & save images, plots, and histograms (strand sim only) (or hit return to skip): ")
    figures = input(
        "Y or any key to display & save images, plots, and histograms (strand sim only) (or hit return to skip): ") or ""

    # Get the current working directory
    cwd = os.getcwd()

    # Define the folder name
    folder_name = "Sim Output"

    # Create the folder path
    folder_path = os.path.join(os.getcwd(), folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if figures:
        display_images(reshaped_images)
        plot_single_cycle_images(reshaped_images, number_of_templates, num_cycles)

        if method_choice == "1":
            plot_histograms(allstrands, number_of_templates, num_cycles)

    param_file = filename + "_params.txt"
    files = input("Y or any key to save parameter and statistics to (or hit return to skip): ")
    if files:
        print("saving ", param_file)
        with open(os.path.join(folder_path, param_file), 'w') as f:
        #with open(param_file, 'w') as f:
            f.write("#454 Sequence simulator with strand simulator for text visualization\n")
            f.write("#Jonathan Rothberg, March 2023\n")
            if method_choice == "1":
                f.write("#tauUV: {}\n".format(tauUV))
                f.write("#uv_time: {}\n".format(uv_time))
                f.write("#tauEX: {}\n".format(tauEX))
                f.write("#ex_time: {}\n".format(ex_time))
                f.write("#p_die: {}\n".format(p_die))
                f.write("#p_dark: {}\n".format(p_dark))
                f.write("#num_strands: {}\n".format(num_strands))
                f.write(f"Number of strands: {num_strands}\n")
            if method_choice == "2":
                f.write("#lag: {}\n".format(lag_percentage))
                f.write("#lead: {}\n".format(lead_percentage))
                f.write("#noise_percentage: {}\n".format(noise_percentage))
                f.write("#death_percentage: {}\n".format(death_percentage))

            f.write("#num_cycles: {}\n".format(num_cycles))
            f.write(f"Number of templates: {number_of_templates}\n")
            f.write(f"Number of templates to process: {number_of_templates}\n")
            f.write(f"Number of training templates: {num_training_templates}\n")
            f.write(f"Template length: {template_length}\n")
            f.write(f"Number of cycles: {num_cycles}\n")
            f.write(f"Window size: {window}\n")
            f.write("#letter_templates: {}\n".format(letter_templates))

            if selected_methods.get("single_image"):
                f.write("called_sequences_single_image: {}\n".format(called_sequences_single_image))
                f.write("Overall Accuracy (Single Image Method)  : {:.2f}%\n".format(accuracy_single_image))

            if selected_methods.get("cnn"):
                f.write("called_sequences_integrated: {}\n".format(called_sequences_integrated))
                f.write("Overall Accuracy (cnn Method)    : {:.2f}%\n".format(accuracy_integrated))

            if selected_methods.get("multipass"):
                f.write("called_sequences_multipass: {}\n".format(called_sequences_multipass))
                f.write("Overall Accuracy (Multipass last pass Method): {:.2f}%\n".format(multilastpassaccuracy))

            if selected_methods.get("kNN"):
                f.write(f"k value for kNN: {k}\n")
                f.write("Called_sequences_kNN: {}\n".format(called_sequences_knn))
                f.write("Overall Accuracy (kNN Method)           : {:.2f}%\n".format(accuracy_knn))

            if selected_methods.get("bidir encoder"):
                f.write("Called_sequences_bidir: {}\n".format(called_sequences_bidir))
                f.write("Overall Accuracy (Bidir Encoder Method) : {:.2f}%\n".format(
                    bidir_accuracy_sum_test / max(num_test_templates, 1)))

            if selected_methods.get("causal transformer"):
                f.write("Called_sequences_causal: {}\n".format(called_sequences_causal))
                f.write("Overall Accuracy (Causal Transformer)   : {:.2f}%\n".format(
                    causal_accuracy_sum_test / max(num_test_templates, 1)))

    if method_choice == "1":
        save_strands = input("Y or any key to save strands (or hit return to skip): ")
        if save_strands:
            strands_file = filename + "_strands.text"
            print("saving ", strands_file)
            with open(os.path.join(folder_path, strands_file), 'w') as f:
                for template_strands in allstrands:
                    for strand in template_strands:
                        f.write("".join(strand) + "\n")
                f.write("----\n")
