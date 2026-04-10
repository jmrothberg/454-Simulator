#independent estimator of lag, lead, death,
import numpy as np
import math
import random
from scipy.optimize import minimize

# Your existing code for calculate_expected_color remains unchanged
def calculate_expected_color(key_colors, cycle, lag, lead, noisel, noisea, noise,  death, key_length):
    expected_color = np.array(key_colors[cycle], dtype=float)

    if cycle > 0:
        cumulative_lag_color = np.zeros(4)
        for prev_cycle in range(cycle):
            lag_multiplier = (1 - lag) ** (cycle - prev_cycle - 1) * (lag * (1 + random.uniform(-1, 1) * noisel))
            cumulative_lag_color += key_colors[prev_cycle] * lag_multiplier

        expected_color[np.argmax(expected_color)] -= np.sum(cumulative_lag_color)
        expected_color += cumulative_lag_color

    if cycle < key_length - 1:
        leading_color = key_colors[cycle + 1] * (lead * (1 + random.uniform(-1, 1) * noisea)) * np.array([1, 1, 1, 1])

        expected_color[np.argmax(expected_color)] -= np.sum(leading_color)
        expected_color += leading_color

    alive_factor = (1 - death * (1 + random.uniform(-1, 1) * noise)) ** cycle

    expected_color *= alive_factor

    return expected_color


def objective_function(params, images, key_colors, key_length, num_templates_to_process):
    lag, lead, noise, death = params
    error = 0

    for cycle in range(key_length):
        for i in range(num_templates_to_process):
            row, col = divmod(i, math.ceil(math.sqrt(num_templates_to_process)))

            spot_color = images[cycle][row][col]
            expected_color = calculate_expected_color(key_colors, cycle, lag, lead, noise, noise, noise, death, key_length)

            error += np.sum((expected_color - np.array(spot_color)) ** 2)

    return error


def estimate_noise_levels(images, key, key_start, key_length, num_templates_to_process, best_lag, best_lead, best_death):
    base_colors = {
        'A': (255, 0, 0, 0),
        'C': (0, 255, 0, 0),
        'G': (0, 0, 255, 0),
        'T': (0, 0, 0, 255)
    }

    key_colors = np.zeros((key_length, 4))

    for i, base in enumerate(key):
        key_colors[i] = np.array(base_colors[base])

    noise_lag_values = []
    noise_lead_values = []
    noise_death_values = []

    epsilon = 1e-8
    for cycle in range(key_length):
        for i in range(num_templates_to_process):
            row, col = divmod(i, math.ceil(math.sqrt(num_templates_to_process)))

            spot_color = images[key_start + cycle][row][col]
            expected_color = calculate_expected_color(key_colors, cycle, best_lag, best_lead, 0, 0, 0, best_death,
                                                      key_length)

            expected_color_noisy = calculate_expected_color(key_colors, cycle, best_lag, best_lead, 0.1, 0.1, 0.1, best_death, key_length)

            noise_lag_values.append(
                np.abs(expected_color[0] - expected_color_noisy[0]) / (np.abs(spot_color[0] - expected_color[0]) + epsilon))
            noise_lead_values.append(
                np.abs(expected_color[1] - expected_color_noisy[1]) / (np.abs(spot_color[1] - expected_color[1]) + epsilon))
            noise_death_values.append(
                np.abs(expected_color[2] - expected_color_noisy[2]) / (np.abs(spot_color[2] - expected_color[2]) + epsilon))

    noise_lag = np.mean(noise_lag_values)
    noise_lead = np.mean(noise_lead_values)
    noise_death = np.mean(noise_death_values)

    return noise_lag, noise_lead, noise_death

def estimate_lag_lead_percentages(images, key, key_start, num_templates_to_process):
    base_colors = {
        'A': (255, 0, 0, 0),
        'C': (0, 255, 0, 0),
        'G': (0, 0, 255, 0),
        'T': (0, 0, 0, 255)
    }

    key_length = len(key)
    key_colors = np.zeros((key_length, 4))

    for i, base in enumerate(key):
        key_colors[i] = np.array(base_colors[base])

    initial_guess = np.array([0.01, 0.01, 0.01, 0.01])  # Initial guess for lag, lead, noise, and death
    bounds = np.array(
        [(0.001, 0.03), (0.001, 0.03), (0.001, 0.1), (0.001, 0.1)])  # Bounds for lag, lead, noise, and death

    result = minimize(
            objective_function,
            initial_guess,
            args=(images, key_colors, key_length, num_templates_to_process),
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 500}
        )

    best_lag, best_lead, best_noise, best_death = result.x
    min_error = result.fun

    noise_lag, noise_lead, noise_death = estimate_noise_levels(images, key, key_start, key_length,
                                                               num_templates_to_process, best_lag, best_lead,
                                                               best_death)

    return best_lag, best_lead, best_death, noise_lag, noise_lead, noise_death, min_error

'''def calculate_difference_statistics(images, key_start, key_length, num_templates_to_process):
    differences = []

    for cycle in range(1, key_length):
        cycle_differences = []

        for i in range(num_templates_to_process - 1):
            row1, col1 = divmod(i, math.ceil(math.sqrt(num_templates_to_process)))
            spot_color1 = images[key_start + cycle][row1][col1]

            for j in range(i + 1, num_templates_to_process):
                row2, col2 = divmod(j, math.ceil(math.sqrt(num_templates_to_process)))
                spot_color2 = images[key_start + cycle][row2][col2]

                cycle_differences.append(np.abs(np.array(spot_color1) - np.array(spot_color2)))

        differences.append(np.mean(cycle_differences, axis=0))

    return differences '''