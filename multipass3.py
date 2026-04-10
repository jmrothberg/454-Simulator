#multipass JMR for 454 April 2nd with iterative passes for accuracy.
import numpy as np
from itertools import product
import math
from scipy.optimize import minimize

# One-hot base vectors (A/C/G/T × 255) — allocated once at import.
_BC = np.array([[255.0, 0.0, 0.0, 0.0],
                [0.0, 255.0, 0.0, 0.0],
                [0.0, 0.0, 255.0, 0.0],
                [0.0, 0.0, 0.0, 255.0]], dtype=np.float64)
_BASE_KEYS = ('A', 'C', 'G', 'T')
_BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

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


def _param_objective(params, ideal_colors_list, observed_spots_list, seq_len_list):
    """L2 objective for scipy.optimize: compares ideal model prediction vs observed signal.
    Supports multiple templates (for joint estimation across all templates)."""
    lag_pct, lead_pct, death_pct, noise_floor = params
    total_error = 0.0
    for ideal, spots, slen in zip(ideal_colors_list, observed_spots_list, seq_len_list):
        for c in range(slen):
            base = ideal[c]
            lag_c = ideal[c - 1] if c > 0 else base
            lead_c = ideal[c + 1] if c < slen - 1 else base
            alive = (1.0 - death_pct) ** c
            expected = base * alive * (1.0 - lag_pct - lead_pct) + \
                       lag_c * lag_pct + lead_c * lead_pct + noise_floor
            diff = expected - spots[c]
            total_error += float(np.dot(diff, diff))
    return total_error


def _optimize_params(ideal_colors_list, observed_spots_list, seq_len_list):
    """Continuous L-BFGS-B optimization of (lag, lead, death, noise_floor)."""
    x0 = [0.01, 0.01, 0.01, 0.0]
    bounds = [(0, 0.3), (0, 0.3), (0, 0.5), (-10, 50)]
    result = minimize(_param_objective, x0,
                      args=(ideal_colors_list, observed_spots_list, seq_len_list),
                      method='L-BFGS-B', bounds=bounds)
    lag, lead, death, nf = result.x
    return {'lag_percent': lag, 'lead_percent': lead, 'death_percent': death, 'noise_floor': nf}


def _seq_to_ideal(seq):
    """Convert a base-letter sequence string to ideal one-hot × 255 color array."""
    return _BC[np.array([_BASE_MAP[b] for b in seq], dtype=np.int32)]


def estimate_lead_lag_death_joint(images, key, num_templates, image_dim):
    """Estimate parameters jointly across ALL templates using the known key region.
    Much more robust than per-template estimation on just 8 key bases."""
    key_len = len(key)
    if key_len < 2:
        return {'lag_percent': 0.0, 'lead_percent': 0.0, 'death_percent': 0.0, 'noise_floor': 0.0}

    ideal = _seq_to_ideal(key)
    ideal_list, spots_list, len_list = [], [], []
    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, image_dim)
        spots = np.array([np.asarray(images[c][row][col], dtype=np.float64).ravel()[:4]
                          for c in range(key_len)])
        ideal_list.append(ideal)
        spots_list.append(spots)
        len_list.append(key_len)

    return _optimize_params(ideal_list, spots_list, len_list)


def estimate_lead_lag_death(images, called_seq, row, col, window_size):
    """Per-template parameter estimation using the called sequence as assumed truth."""
    seq_len = len(called_seq)
    if seq_len < 2:
        return {'lag_percent': 0.0, 'lead_percent': 0.0, 'death_percent': 0.0, 'noise_floor': 0.0}

    ideal = _seq_to_ideal(called_seq)
    spots = np.array([np.asarray(images[c][row][col], dtype=np.float64).ravel()[:4]
                      for c in range(seq_len)])
    return _optimize_params([ideal], [spots], [seq_len])


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

    # Pass 1: joint estimation across ALL templates using the key region (robust)
    joint_params = estimate_lead_lag_death_joint(images, key, num_templates, image_dim)
    print(f"Joint key estimate: Lag: {joint_params['lag_percent']:.4f}, Lead: {joint_params['lead_percent']:.4f}, "
          f"Death: {joint_params['death_percent']:.4f}, Noise: {joint_params['noise_floor']:.2f}")

    for seq_idx in range(num_templates):
        row, col = divmod(seq_idx, image_dim)

        # Precompute all spot colors for this template — avoids re-reading from nested lists
        spot_array = np.array([np.asarray(images[c][row][col], dtype=np.float64).ravel()[:4]
                               for c in range(num_cycles)])  # (num_cycles, 4)

        assembled_seq_passes = []
        current_key = key

        for pass_idx in range(num_passes):
            if pass_idx == 0:
                # Use the robust joint estimate for the first pass
                params = joint_params
            else:
                # Subsequent passes: refine per-template using the full called sequence
                params = estimate_lead_lag_death(images, current_key, row, col, window_size)

            lag_pct = params['lag_percent']
            lead_pct = params['lead_percent']
            death_pct = params['death_percent']
            noise_floor = params['noise_floor']

            print(f"Pass {pass_idx + 1}: Lag: {lag_pct:.4f}, Lead: {lead_pct:.4f}, Death: {death_pct:.4f}, Noise: {noise_floor:.2f}")

            # reduction_factor is constant (1 - lag% - lead%) because all base vectors are
            # one-hot × 255: peak=255, sum(lag_color)=255, so sum(lag*lag%)/peak = lag%.
            reduction = 1.0 - lag_pct - lead_pct
            death_base = 1.0 - death_pct

            # Consensus voting: each window votes for ALL its positions, weighted by
            # confidence.  Normalizing to unit vectors before comparison focuses on
            # which-base (direction) rather than how-much-signal (amplitude), making
            # the method robust to imprecise death estimates at late cycles.
            base_votes = np.zeros((num_cycles, 4), dtype=np.float64)

            for cycle in range(start_cycle, end_cycle_exclusive):
                spots = spot_array[cycle:cycle + window_size]  # (ws, 4)
                alive = death_base ** np.arange(cycle, cycle + window_size, dtype=np.float64)

                expected = combo_colors * (alive[None, :, None] * reduction) + \
                           lag_colors * lag_pct + lead_colors * lead_pct + noise_floor

                # Normalize both to unit vectors for direction-based matching
                s_norm = np.linalg.norm(spots, axis=1, keepdims=True)
                s_norm = np.maximum(s_norm, 1e-12)
                e_norm = np.linalg.norm(expected, axis=2, keepdims=True)
                e_norm = np.maximum(e_norm, 1e-12)

                diff = (expected / e_norm) - (spots / s_norm)[None, :, :]
                errors = np.einsum('ijk,ijk->i', diff, diff)

                best_idx = int(np.argmin(errors))
                weight = 1.0 / (1.0 + errors[best_idx])
                best_combo = all_combos[best_idx]
                for w in range(window_size):
                    base_votes[cycle + w, best_combo[w]] += weight

            # Resolve consensus: same position range as the original center-only approach
            call_start = len(key)
            call_end = end_cycle_exclusive - 1 + middle_index + 1
            assembled_seq = [_BASE_KEYS[int(np.argmax(base_votes[pos]))]
                             for pos in range(call_start, call_end)]

            called_bases = key + "".join(assembled_seq)
            assembled_seq_passes.append(called_bases)
            # Refit using the full latest call for next pass
            current_key = called_bases

            print(f"MultiPass {pass_idx + 1}: {called_bases}")

        called_bases_passes.append(assembled_seq_passes)

    return called_bases_passes