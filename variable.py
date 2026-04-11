#variable base caller
# We have ESTIMATES of the lag and lead, but don't know it exactly, but this helps us see what of the possible 5 base combinations best fits the window
# We slide the window along the images, keeping the center base to try to determine what the original template seqeunce was. We can't cheat, the gound truth is only to
# know how long the sequene is, and later to see how accurate our base calling is.

import math
from itertools import product

import numpy as np

# One-hot base vectors (A/C/G/T × 255) — allocated once at import.
_BC = np.array([[255.0, 0.0, 0.0, 0.0],
                [0.0, 255.0, 0.0, 0.0],
                [0.0, 0.0, 255.0, 0.0],
                [0.0, 0.0, 0.0, 255.0]], dtype=np.float64)
_BASE_KEYS = ('A', 'C', 'G', 'T')

_combo_cache = {}

def _get_combos(ws):
    if ws not in _combo_cache:
        idx = np.array(list(product(range(4), repeat=ws)), dtype=np.int32)
        colors = _BC[idx]
        lag = np.empty_like(colors)
        lag[:, 0]  = colors[:, 0]
        lag[:, 1:] = colors[:, :-1]
        lead = np.empty_like(colors)
        lead[:, -1]  = colors[:, -1]
        lead[:, :-1] = colors[:, 1:]
        _combo_cache[ws] = (idx, colors, lag, lead)
    return _combo_cache[ws]


def base_calling_uncertain_lag_lead(images, num_cycles, lag_percent, lead_percent, death_percent, num_templates, window_size):
    print ("base_calling_uncertain_lag_lead")

    middle_index = window_size // 2
    called_bases = [''] * num_templates
    all_combos, combo_colors, lag_colors, lead_colors = _get_combos(window_size)
    image_dim = math.ceil(math.sqrt(num_templates))

    reduction = 1.0 - lag_percent - lead_percent
    death_base = 1.0 - death_percent

    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, image_dim)

        spot_array = np.array([np.asarray(images[c][row][col], dtype=np.float64).ravel()[:4]
                               for c in range(num_cycles)])

        assembled_seq = []
        for cycle in range(num_cycles - window_size + 1):
            spots = spot_array[cycle:cycle + window_size]
            alive = death_base ** np.arange(cycle, cycle + window_size, dtype=np.float64)

            # All signal (main + lag + lead) comes from alive strands
            expected = (combo_colors * reduction + lag_colors * lag_percent + lead_colors * lead_percent) \
                       * alive[None, :, None]
            diff = expected - spots[None, :, :]
            errors = np.einsum('ijk,ijk->i', diff, diff)

            best_idx = int(np.argmin(errors))
            best_combo = all_combos[best_idx]

            #print("For cycle, min_error, best_window_seq:", cycle, min_error, best_window_seq_letters)
            if cycle == 0:
                assembled_seq.extend([_BASE_KEYS[base_idx] for base_idx in best_combo[:middle_index + 1]])
            else:
                assembled_seq.append(_BASE_KEYS[best_combo[middle_index]])

        called_bases[seq_idx] = ''.join(assembled_seq)

    return called_bases
