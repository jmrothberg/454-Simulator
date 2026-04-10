#multipass JMR for 454 April 2nd with iterative passes for accuracy.
import numpy as np
from itertools import product
import math
from joblib import Parallel, delayed

def estimate_death_percent(images, key, row, col):
    base_colors = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    death_percent_sum = 0
    count = 0

    for i in range(len(key) - 1):
        spot_color1 = images[i][row][col]
        spot_color2 = images[i + 1][row][col]

        base_idx1 = base_colors[key[i]]
        base_idx2 = base_colors[key[i + 1]]

        if np.argmax(spot_color1) == base_idx1 and np.argmax(spot_color2) == base_idx2:
            death_percent_sum += 1 - (spot_color2[base_idx2] / spot_color1[base_idx1])
            count += 1

    death_percent = death_percent_sum / count
    return death_percent

def estimate_lead_lag_death_worker(images, key, row, col, window_size, lead_lag_combination):

    death_percent = estimate_death_percent(images, key, row, col)

    lead_percent, lag_percent = lead_lag_combination
    total_error = 0

    for cycle in range(len(key) - window_size + 1):
        current_cycle = cycle
        spot_color = images[current_cycle][row][col]
        base_color = np.array(spot_color)

        if cycle > 0:
            lagging_base_color = images[current_cycle - 1][row][col]
        else:
            lagging_base_color = base_color

        if cycle < len(key) - 1:
            leading_base_color = images[current_cycle + 1][row][col]
        else:
            leading_base_color = base_color

        reduction_factor = 1 - (lagging_base_color * lag_percent) / np.sum(base_color)
        reduction_factor -= (leading_base_color * lead_percent) / np.sum(base_color)

        base_color *= (1 - death_percent)

        expected_color = base_color * reduction_factor
        expected_color += lagging_base_color * lag_percent
        expected_color += leading_base_color * lead_percent

        # Implement the improved error calculation
        error = np.sum(np.abs(expected_color - spot_color))
        total_error += error

    avg_error = total_error / (len(key) - window_size + 1)

    return lead_lag_combination, avg_error

def estimate_lead_lag_death(images, key, row, col, window_size):  #uses estimate_lead_lag_death_worker

    combinations = list(product(np.linspace(0, 0.2, 21), repeat=2))
    results = Parallel(n_jobs=-1)(delayed(estimate_lead_lag_death_worker)(images, key, row, col, window_size, comb) for comb in combinations)

    best_lead_lag_combination, min_error = min(results, key=lambda x: x[1])

    death_percent = estimate_death_percent(images, key, row, col)

    return {
    'lead_percent': best_lead_lag_combination[0],
    'lag_percent': best_lead_lag_combination[1],
    'death_percent': death_percent,
    }

def base_calling_multipass(images, num_cycles, key, num_templates, window_size, num_passes): # uses estimate_lead_lag_death
    base_colors = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    base_color_key = {0: (255.0, 0.0, 0.0, 0.0), 1: (0.0, 255.0, 0.0, 0.0), 2: (0.0, 0.0, 255.0, 0.0), 3: (0.0, 0.0, 0.0, 255.0)}

    called_bases_passes = []

    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, math.ceil(math.sqrt(num_templates)))

        assembled_seq_passes = []
        current_key = key  # Initialize current_key with the known key

        for pass_idx in range(num_passes):
            assembled_seq = []

            # Use the current_key to estimate lead, lag, and death percentages
            lead_lag_death = estimate_lead_lag_death(images, current_key, row, col, window_size)

            print(f"Pass {pass_idx + 1}: Lag: {lead_lag_death['lag_percent']:.2f}, Lead: {lead_lag_death['lead_percent']:.2f}, Death: {lead_lag_death['death_percent']:.2f}")

            # Use the estimated lead, lag, and death percentages to call the unknown sequence

            start_cycle = len(current_key) - window_size + (window_size // 2) + 1 if pass_idx == 0 else len(key) - window_size + (window_size // 2) + 1

            for cycle in range(start_cycle, num_cycles - window_size + 1):

                min_error = float('inf')
                best_window_seq = ""

                for base_combination in product(range(4), repeat=window_size):
                    error = 0

                    for w in range(window_size):
                        current_cycle = cycle + w
                        spot_color = images[current_cycle][row][col]
                        base_color = np.array(base_color_key[base_combination[w]])

                        if w > 0:
                            lagging_base_color = np.array(base_color_key[base_combination[w - 1]])
                        else:
                            lagging_base_color = base_color

                        if w < window_size - 1:
                            leading_base_color = np.array(base_color_key[base_combination[w + 1]])
                        else:
                            leading_base_color = base_color

                        reduction_factor = 1 - np.sum(lagging_base_color * lead_lag_death['lag_percent']) / base_color[np.argmax(base_color)]
                        reduction_factor -= np.sum(leading_base_color * lead_lag_death['lead_percent']) / base_color[np.argmax(base_color)]

                        alive_factor = (1 - lead_lag_death['death_percent']) ** current_cycle
                        base_color *= alive_factor

                        expected_color = base_color * reduction_factor
                        expected_color += lagging_base_color * lead_lag_death['lag_percent']
                        expected_color += leading_base_color * lead_lag_death['lead_percent']

                        error += np.sum(np.abs(expected_color - spot_color))

                    if error < min_error:
                        min_error = error
                        best_window_seq = base_combination

                middle_index = window_size // 2
                assembled_seq.append(list(base_colors.keys())[best_window_seq[middle_index]])

            called_bases = key + "".join(assembled_seq)

            assembled_seq_passes.append(called_bases)

            if pass_idx > 0:
                current_key = key + assembled_seq_passes[pass_idx - 1][len(key):]

            print(f"MultiPass {pass_idx + 1}: {called_bases}")

        called_bases_passes.append(assembled_seq_passes)

    return called_bases_passes