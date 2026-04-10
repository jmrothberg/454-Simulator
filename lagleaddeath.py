# Independent estimator of lag, lead, death (deterministic forward model for stable fitting).
import math

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-9


def calculate_expected_color(
    key_colors, cycle, lag, lead, noisel, noisea, death_noise, death, key_length
):
    """
    Deterministic SBS-style color at one cycle: nominal one-hot key color with
    exponential lag from prior cycles, lead from next cycle, and per-cycle survival (death).

    noisel / noisea / death_noise are small nonnegative multiplicative scales (no RNG) so
    L-BFGS-B optimizes a fixed objective surface.
    """
    expected_color = np.array(key_colors[cycle], dtype=np.float64, copy=True)
    lag_amp = 1.0 + float(noisel)
    lead_amp = 1.0 + float(noisea)
    death_amp = 1.0 + float(death_noise)

    if cycle > 0:
        cumulative_lag_color = np.zeros(4, dtype=np.float64)
        for prev_cycle in range(cycle):
            lag_multiplier = (
                (1.0 - lag) ** (cycle - prev_cycle - 1) * lag * lag_amp
            )
            cumulative_lag_color += key_colors[prev_cycle] * lag_multiplier

        peak = int(np.argmax(expected_color))
        expected_color[peak] -= np.sum(cumulative_lag_color)
        expected_color += cumulative_lag_color

    if cycle < key_length - 1:
        leading_color = key_colors[cycle + 1] * (lead * lead_amp)
        peak = int(np.argmax(expected_color))
        expected_color[peak] -= np.sum(leading_color)
        expected_color += leading_color

    # Survival: death in (0,1); death_amp nudges effective loss (kept small by bounds)
    alive_factor = (1.0 - death * death_amp) ** cycle
    expected_color *= alive_factor

    return expected_color


def objective_function(
    params, images, key_colors, key_length, num_templates_to_process, key_start
):
    lag, lead, noise, death = params
    error = 0.0
    # Single noise scale applied consistently (same as previous API: one learned perturbation)
    n = float(noise)

    for cycle in range(key_length):
        for i in range(num_templates_to_process):
            row, col = divmod(i, math.ceil(math.sqrt(num_templates_to_process)))

            spot_color = np.asarray(images[key_start + cycle][row][col], dtype=np.float64).ravel()[:4]
            expected_color = calculate_expected_color(
                key_colors,
                cycle,
                lag,
                lead,
                n,
                n,
                n,
                death,
                key_length,
            )

            diff = expected_color - spot_color
            error += float(np.dot(diff, diff))

    return error


def estimate_noise_levels(
    images,
    key,
    key_start,
    key_length,
    num_templates_to_process,
    best_lag,
    best_lead,
    best_death,
    bump=0.05,
):
    """
    Local sensitivity: how much the predicted vector moves when lag, lead, or death_noise
    bumps vs residual norm to data (not tied to arbitrary channel indices).
    """
    base_colors = {
        "A": (255, 0, 0, 0),
        "C": (0, 255, 0, 0),
        "G": (0, 0, 255, 0),
        "T": (0, 0, 0, 255),
    }

    key_colors = np.zeros((key_length, 4), dtype=np.float64)
    for i, base in enumerate(key):
        key_colors[i] = np.array(base_colors[base], dtype=np.float64)

    noise_lag_values = []
    noise_lead_values = []
    noise_death_values = []

    for cycle in range(key_length):
        for i in range(num_templates_to_process):
            row, col = divmod(i, math.ceil(math.sqrt(num_templates_to_process)))

            spot_color = np.asarray(images[key_start + cycle][row][col], dtype=np.float64).ravel()[:4]
            expected_color = calculate_expected_color(
                key_colors, cycle, best_lag, best_lead, 0.0, 0.0, 0.0, best_death, key_length
            )
            exp_lag = calculate_expected_color(
                key_colors, cycle, best_lag, best_lead, bump, 0.0, 0.0, best_death, key_length
            )
            exp_lead = calculate_expected_color(
                key_colors, cycle, best_lag, best_lead, 0.0, bump, 0.0, best_death, key_length
            )
            exp_death = calculate_expected_color(
                key_colors, cycle, best_lag, best_lead, 0.0, 0.0, bump, best_death, key_length
            )

            residual_norm = np.linalg.norm(spot_color - expected_color) + _EPS
            noise_lag_values.append(float(np.linalg.norm(exp_lag - expected_color) / residual_norm))
            noise_lead_values.append(float(np.linalg.norm(exp_lead - expected_color) / residual_norm))
            noise_death_values.append(float(np.linalg.norm(exp_death - expected_color) / residual_norm))

    noise_lag = float(np.mean(noise_lag_values))
    noise_lead = float(np.mean(noise_lead_values))
    noise_death = float(np.mean(noise_death_values))

    return noise_lag, noise_lead, noise_death


def estimate_lag_lead_percentages(images, key, key_start, num_templates_to_process):
    base_colors = {
        "A": (255, 0, 0, 0),
        "C": (0, 255, 0, 0),
        "G": (0, 0, 255, 0),
        "T": (0, 0, 0, 255),
    }

    key_length = len(key)
    key_colors = np.zeros((key_length, 4), dtype=np.float64)

    for i, base in enumerate(key):
        key_colors[i] = np.array(base_colors[base], dtype=np.float64)

    initial_guess = np.array([0.01, 0.01, 0.01, 0.01])
    bounds = np.array(
        [(0.001, 0.03), (0.001, 0.03), (0.001, 0.1), (0.001, 0.1)]
    )

    result = minimize(
        objective_function,
        initial_guess,
        args=(images, key_colors, key_length, num_templates_to_process, key_start),
        method="L-BFGS-B",
        bounds=list(zip(bounds[:, 0], bounds[:, 1])),
        options={"maxiter": 500},
    )

    best_lag, best_lead, best_noise, best_death = result.x
    min_error = result.fun

    noise_lag, noise_lead, noise_death = estimate_noise_levels(
        images,
        key,
        key_start,
        key_length,
        num_templates_to_process,
        best_lag,
        best_lead,
        best_death,
    )

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
