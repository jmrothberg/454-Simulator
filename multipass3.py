#multipass JMR for 454 April 2nd with iterative passes for accuracy.
import numpy as np
from itertools import product
import math

# One-hot base vectors (A/C/G/T × 255) — allocated once at import.
_BC = np.array([[255.0, 0.0, 0.0, 0.0],
                [0.0, 255.0, 0.0, 0.0],
                [0.0, 0.0, 255.0, 0.0],
                [0.0, 0.0, 0.0, 255.0]], dtype=np.float64)
_BASE_KEYS = ('A', 'C', 'G', 'T')

# Cached per window_size so the 4^ws table is built once per session.
_combo_cache = {}

def _get_combos(ws):
    """Return (indices, base_colors, lag_colors, lead_colors) arrays for all 4^ws combinations."""
    if ws not in _combo_cache:
        idx = np.array(list(product(range(4), repeat=ws)), dtype=np.int32)  # (4^ws, ws)
        colors = _BC[idx]                    # (N, ws, 4)
        lag = np.empty_like(colors)
        lag[:, 0]  = colors[:, 0]            # w=0: lag neighbor is self
        lag[:, 1:] = colors[:, :-1]
        lead = np.empty_like(colors)
        lead[:, -1]  = colors[:, -1]         # last w: lead neighbor is self
        lead[:, :-1] = colors[:, 1:]
        _combo_cache[ws] = (idx, colors, lag, lead)
    return _combo_cache[ws]


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


def estimate_lead_lag_death(images, key, row, col, window_size):
    """Grid-search for best (lead, lag) — runs in-process (faster than joblib for this tiny workload)."""
    death_percent = estimate_death_percent(images, key, row, col)
    key_len = len(key)
    n_windows = key_len - window_size + 1
    if n_windows <= 0:
        return {'lead_percent': 0.0, 'lag_percent': 0.0, 'death_percent': death_percent}

    # Precompute spot colors for key cycles once
    spots = np.array([np.asarray(images[c][row][col], dtype=np.float64).ravel()[:4] for c in range(key_len)])
    alive_mult = 1.0 - death_percent

    grid = np.linspace(0, 0.2, 21)
    best_error = float('inf')
    best_lead = 0.0
    best_lag = 0.0

    for lead_pct in grid:
        for lag_pct in grid:
            total_error = 0.0
            for c in range(n_windows):
                spot = spots[c]
                base = spot.copy()
                lag_c = spots[c - 1] if c > 0 else spot
                lead_c = spots[c + 1] if c < key_len - 1 else spot

                den = max(float(np.sum(base)), 1e-12)
                reduction = 1.0 - (lag_c * lag_pct) / den - (lead_c * lead_pct) / den
                base *= alive_mult
                expected = base * reduction + lag_c * lag_pct + lead_c * lead_pct
                diff = expected - spot
                total_error += float(np.sum(np.abs(diff)))

            if total_error < best_error:
                best_error = total_error
                best_lead = lead_pct
                best_lag = lag_pct

    return {'lead_percent': best_lead, 'lag_percent': best_lag, 'death_percent': death_percent}


def base_calling_multipass(images, num_cycles, key, num_templates, window_size, num_passes):
    called_bases_passes = []
    middle_index = window_size // 2
    # First window whose center sits on the first base after the known key prefix
    start_cycle = len(key) - window_size + middle_index + 1
    start_cycle = max(0, start_cycle)
    # cycle is the window START; window covers [cycle, cycle+window_size-1], so cap at num_cycles-window_size
    end_cycle_exclusive = num_cycles - window_size + 1

    if len(key) < window_size or start_cycle >= end_cycle_exclusive:
        return [[key] * num_passes for _ in range(num_templates)]

    # Precompute all 4^ws combo/lag/lead color arrays once
    all_combos, combo_colors, lag_colors, lead_colors = _get_combos(window_size)
    image_dim = math.ceil(math.sqrt(num_templates))

    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, image_dim)

        # Precompute all spot colors for this template — avoids re-reading from nested lists
        spot_array = np.array([np.asarray(images[c][row][col], dtype=np.float64).ravel()[:4]
                               for c in range(num_cycles)])  # (num_cycles, 4)

        assembled_seq_passes = []
        current_key = key

        for pass_idx in range(num_passes):
            lead_lag_death = estimate_lead_lag_death(images, current_key, row, col, window_size)
            lag_pct = lead_lag_death['lag_percent']
            lead_pct = lead_lag_death['lead_percent']
            death_pct = lead_lag_death['death_percent']

            print(f"Pass {pass_idx + 1}: Lag: {lag_pct:.2f}, Lead: {lead_pct:.2f}, Death: {death_pct:.2f}")

            # reduction_factor is constant (1 - lag% - lead%) because all base vectors are
            # one-hot × 255: peak=255, sum(lag_color)=255, so sum(lag*lag%)/peak = lag%.
            reduction = 1.0 - lag_pct - lead_pct
            death_base = 1.0 - death_pct

            assembled_seq = []
            for cycle in range(start_cycle, end_cycle_exclusive):
                spots = spot_array[cycle:cycle + window_size]  # (ws, 4)
                alive = death_base ** np.arange(cycle, cycle + window_size, dtype=np.float64)  # (ws,)

                # Vectorized expected color for ALL combos at once: (N_combos, ws, 4)
                expected = combo_colors * (alive[None, :, None] * reduction) + \
                           lag_colors * lag_pct + lead_colors * lead_pct

                diff = expected - spots[None, :, :]  # (N_combos, ws, 4)
                errors = np.einsum('ijk,ijk->i', diff, diff)  # sum-of-squared-error per combo

                best_idx = int(np.argmin(errors))
                assembled_seq.append(_BASE_KEYS[all_combos[best_idx, middle_index]])

            called_bases = key + "".join(assembled_seq)
            assembled_seq_passes.append(called_bases)
            # BUGFIX: refit lag/lead/death on the full latest call (was stuck on `key` after pass 0).
            current_key = called_bases

            print(f"MultiPass {pass_idx + 1}: {called_bases}")

        called_bases_passes.append(assembled_seq_passes)

    return called_bases_passes