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
        spot_color1 = np.asarray(images[i][row][col])[:4]
        spot_color2 = np.asarray(images[i + 1][row][col])[:4]

        base_idx1 = base_colors[key[i]]
        base_idx2 = base_colors[key[i + 1]]

        if np.argmax(spot_color1) == base_idx1 and np.argmax(spot_color2) == base_idx2:
            d1 = float(spot_color1[base_idx1])
            if d1 <= 1e-12:
                continue
            death_percent_sum += 1 - (spot_color2[base_idx2] / d1)
            count += 1

    # BUGFIX: no matching transitions → avoid div by zero
    if count == 0:
        return 0.0
    return death_percent_sum / count

def estimate_lead_lag_death_worker(images, key, row, col, window_size, lead_lag_combination):

    death_percent = estimate_death_percent(images, key, row, col)

    lead_percent, lag_percent = lead_lag_combination
    total_error = 0

    for cycle in range(len(key) - window_size + 1):
        current_cycle = cycle
        spot_color = np.asarray(images[current_cycle][row][col], dtype=np.float64).ravel()[:4]
        base_color = np.array(spot_color)

        if cycle > 0:
            lagging_base_color = np.asarray(images[current_cycle - 1][row][col])[:4]
        else:
            lagging_base_color = base_color

        if cycle < len(key) - 1:
            leading_base_color = np.asarray(images[current_cycle + 1][row][col])[:4]
        else:
            leading_base_color = base_color

        den = float(np.sum(base_color))
        if den < 1e-12:
            den = 1e-12
        reduction_factor = 1 - (lagging_base_color * lag_percent) / den
        reduction_factor -= (leading_base_color * lead_percent) / den

        base_color *= (1 - death_percent)

        expected_color = base_color * reduction_factor
        expected_color += lagging_base_color * lag_percent
        expected_color += leading_base_color * lead_percent

        # Implement the improved error calculation
        error = np.sum(np.abs(expected_color - spot_color))
        total_error += error

    denom = len(key) - window_size + 1
    if denom <= 0:
        return lead_lag_combination, float('inf')
    avg_error = total_error / denom

    return lead_lag_combination, avg_error

def estimate_lead_lag_death(images, key, row, col, window_size):  #uses estimate_lead_lag_death_worker

    combinations = list(product(np.linspace(0, 0.2, 21), repeat=2))
    results = Parallel(n_jobs=-1)(delayed(estimate_lead_lag_death_worker)(images, key, row, col, window_size, comb) for comb in combinations)

    finite = [r for r in results if r[1] < float('inf')]
    best_lead_lag_combination, min_error = min(finite if finite else results, key=lambda x: x[1])

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
    middle_index = window_size // 2
    # First window whose center sits on the first base after the known key prefix
    start_cycle = len(key) - window_size + middle_index + 1
    start_cycle = max(0, start_cycle)
    # cycle is the window START; window covers [cycle, cycle+window_size-1], so cap at num_cycles-window_size
    end_cycle_exclusive = num_cycles - window_size + 1

    if len(key) < window_size or start_cycle >= end_cycle_exclusive:
        return [[key] * num_passes for _ in range(num_templates)]

    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, math.ceil(math.sqrt(num_templates)))

        assembled_seq_passes = []
        current_key = key

        for pass_idx in range(num_passes):
            assembled_seq = []

            # Use the current_key to estimate lead, lag, and death percentages
            lead_lag_death = estimate_lead_lag_death(images, current_key, row, col, window_size)

            print(f"Pass {pass_idx + 1}: Lag: {lead_lag_death['lag_percent']:.2f}, Lead: {lead_lag_death['lead_percent']:.2f}, Death: {lead_lag_death['death_percent']:.2f}")

            # Use the estimated lead, lag, and death percentages to call the unknown sequence

            for cycle in range(start_cycle, end_cycle_exclusive):

                min_error = float('inf')
                best_window_seq = ""

                for base_combination in product(range(4), repeat=window_size):
                    error = 0

                    for w in range(window_size):
                        current_cycle = cycle + w
                        spot_color = np.asarray(images[current_cycle][row][col], dtype=np.float64).ravel()[:4]
                        base_color = np.array(base_color_key[base_combination[w]])

                        if w > 0:
                            lagging_base_color = np.array(base_color_key[base_combination[w - 1]])
                        else:
                            lagging_base_color = base_color

                        if w < window_size - 1:
                            leading_base_color = np.array(base_color_key[base_combination[w + 1]])
                        else:
                            leading_base_color = base_color

                        peak = float(base_color[np.argmax(base_color)])
                        if peak < 1e-12:
                            peak = 1e-12
                        reduction_factor = 1 - np.sum(lagging_base_color * lead_lag_death['lag_percent']) / peak
                        reduction_factor -= np.sum(leading_base_color * lead_lag_death['lead_percent']) / peak

                        alive_factor = (1 - lead_lag_death['death_percent']) ** current_cycle
                        base_color *= alive_factor

                        expected_color = base_color * reduction_factor
                        expected_color += lagging_base_color * lead_lag_death['lag_percent']
                        expected_color += leading_base_color * lead_lag_death['lead_percent']

                        # Least-squares match to Gaussian-ish noise (consistent with integrated2 / variable callers)
                        diff = expected_color - spot_color
                        error += float(np.dot(diff, diff))

                    if error < min_error:
                        min_error = error
                        best_window_seq = base_combination

                assembled_seq.append(list(base_colors.keys())[best_window_seq[middle_index]])

            called_bases = key + "".join(assembled_seq)

            assembled_seq_passes.append(called_bases)
            # BUGFIX: refit lag/lead/death on the full latest call (was stuck on `key` after pass 0).
            current_key = called_bases

            print(f"MultiPass {pass_idx + 1}: {called_bases}")

        called_bases_passes.append(assembled_seq_passes)

    return called_bases_passes