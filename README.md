# 454 Sequencing Simulator

Python simulations of **sequencing by synthesis** in the spirit of 454-style pyrosequencing / E‑wave chemistry with Lightning terminators: strand-level dynamics, UV and excitation timing, dark bases and strand loss, image-style four-color (A/C/G/T) channels, and several **base callers** (classical window search, kNN, CNN, a bidirectional window encoder, and a real autoregressive (causal/GPT-style) transformer in TensorFlow).

Original research and code by **Jonathan M. Rothberg** (see file headers for dates and notes).

---

## What this repo does

1. **Simulate** many DNA strands (or a simpler synthetic image generator) cycle by cycle and record **four dye channels** per template “spot” per cycle—like a coarse sequencing image or pyrogram.
2. **Call** (decode) DNA from those images using different algorithms and **compare** calls to the known template to see accuracy.

You do **not** need to use every caller; pick what you want when `MultiSim2` asks.

---

## Two entry points

### `454Sim13.py` — one template, many strands, simple plot

- Interactive prompts: timing (`tauUV`, `uv_time`, `tauEX`, `ex_time`), strand death / dark-base rates, one DNA template, number of cycles.
- Simulates **one** template with **many** strands (default 1000), aggregates **terminal dye** counts per cycle, plots a **stacked bar chart**, and applies a **simple per-cycle** base-calling rule.
- Good when you want a **single pyrogram-style figure** and strand text dumps without the multi-template grid or ML models.

```bash
python 454Sim13.py
```

### `MultiSim2.py` — many templates, choice of simulator, many callers

- Generates **many random templates** that all share a **known key** prefix (for calibration-style callers).
- You choose **how images are built** (see below), then **which analysis methods** to run (by index).
- Prints **called sequences vs original** and **accuracy** per method; optional saves under `Sim Output/`.

```bash
python MultiSim2.py
```

Run from the **project root** so imports and the `Sim Models/` folder resolve correctly.

---

## MultiSim2: how images are built (method 1 vs 2)

When the script asks **Choose method: (1) strand-based simulator, (2) simple lead lag noise simulator**:

| Choice | What it means |
|--------|----------------|
| **1 — Strand-based** | Full **strand-level** simulation (same physics family as `454Sim13`) for **each** template, with many strands per spot. Counts blocked terminators per cycle (including dye **right before** a strand is marked dead so early death still contributes signal). Scales counts so channel sums stay in a range comparable to the synthetic path. |
| **2 — Synthetic** | No strand lists. Each cycle uses the **ideal** A/C/G/T corner vector for that template position, then applies **lag**, **lead**, **noise**, and **death** (fading) in closed form. Faster and smoother; good for testing callers without strand stochasticity. |

**Suggested first run:** start with **2** if you only want to see callers behave; use **1** when you care about strand-level effects (death, dark bases, lag markers, etc.).

---

## MultiSim2: analysis methods (menu indices)

When prompted for **indices of the methods** (space-separated), the menu is:

| # | Name | What it does |
|---|------|----------------|
| 1 | **single_image** | Per cycle and per template spot: read the four channels. If signal is too weak or ambiguous, call **`N`**; else pick the strongest channel vs ideal A/C/G/T corners. |
| 2 | **multipass** | **SciPy L-BFGS-B** optimizes lag, lead, death, and a noise-floor from the known key using **ideal one-hot colors** (not observed signal). Pass 1 uses a **joint estimate across all templates**; later passes refine per-template. Vectorized **window search** (numpy `einsum`) over all 4^ws combos with **consensus voting** across overlapping windows and **normalized (direction-based) comparison**. Implemented in `multipass3.py`. |
| 3 | **kNN** | **k-nearest neighbors** with **StandardScaler** + **distance weighting**: trains on windows of color vectors → middle base at each slide. Batched inference. Can save/load models in `Sim Models/`. |
| 4 | **cnn** | **1D CNN** with **per-position prediction** (4 bases per window position, not 1024 combo classes). Three `Conv1D(padding='same')` layers preserve window dimension; per-position softmax output. **EarlyStopping** (patience 6). Saves `.h5` weights + `.pkl` metadata in `Sim Models/`. |
| 5 | **bidir encoder** | **Bidirectional Window Encoder (BERT-style, NOT autoregressive)**. Slides a fixed window (default 5 cycles) with **full bidirectional self-attention** — every position sees every other. 3 encoder blocks (64-dim, 4 heads, 128-unit FFN), per-position softmax; takes the **center position** as the call. **Auto-checkpoint**: warm-starts from `_auto_transformer_w{ws}.h5`. Implemented in `transformer8.py`. |
| 6 | **causal transformer** | **Real autoregressive (GPT-style) transformer**. Processes the **full read** (all cycles) in one forward pass. **Causal masking**: position *t* can only attend to cycles 0..*t* (past + self). 4 causal blocks (64-dim, 4 heads, 128-unit FFN), **sinusoidal positional encoding**, per-position softmax. **Auto-checkpoint**: saves to `_auto_causal_transformer_c{cycles}.h5`; warm-starts with lower LR. Implemented in `causal_transformer.py`. |
| 7 | **estimate lag, lead, noise, death** | Runs **`lagleaddeath.estimate_lag_lead_percentages`**: **L-BFGS-B** fits global lag, lead, death, and a small coupling scale to match images to the **key** (deterministic forward model—no RNG inside the objective). Prints **sensitivity indices** (how much the predicted vector moves under small lag/lead/death bumps, relative to fit residual)—not literal “noise %.” |

**Default run:** press **Enter** (or type `all`) to run all 7 methods automatically. All ML callers train fresh (or warm-start from their checkpoints) without interactive prompts.

---

## Other Python modules (reference)

| File | Role |
|------|------|
| `454Sim13.py` | Single-template strand simulator + pyrogram-style plot + simple caller. |
| `MultiSim2.py` | Main pipeline: template generation, image construction, method dispatch, accuracy reporting, optional plots/saves. |
| `multipass3.py` | Multipass caller: scipy L-BFGS-B parameter estimation (joint across templates), vectorized window search with consensus voting. **This is what `MultiSim2` imports.** |
| `lagleaddeath.py` | SciPy **L-BFGS-B** fit of lag, lead, death + scale; **deterministic** color model for stable optimization. |
| `knn_caller4.py` | kNN training/inference and optional `Sim Models/*.pkl` persistence. |
| `cnn_caller.py` | CNN training/inference; weights `.h5` + encoders `.pkl` in `Sim Models/`. |
| `transformer8.py` | **Bidirectional Window Encoder** (BERT-style) training/inference; `.h5` models in `Sim Models/`. |
| `causal_transformer.py` | **Autoregressive (GPT-style) Transformer** with causal masking and sinusoidal positional encoding; `.h5` checkpoints in `Sim Models/`. |
| `integrated2.py` | Integrated window caller + a **second** multipass implementation (for direct import); `MultiSim2` does **not** use this file for multipass. |
| `variable.py` | Window caller when **you** supply lag/lead/death; exhaustive window search. Not on the default `MultiSim2` menu—import if you need it. |

---

## Folders

| Folder | Purpose |
|--------|---------|
| **`Sim Models/`** | Created automatically. Stores saved **kNN** pickles, **CNN** `.h5`/`.pkl`, **bidir encoder** `.h5`, **causal transformer** `.h5`, etc. Safe to delete only if you are fine retraining. |
| **`Sim Output/`** | Optional outputs from `MultiSim2` (parameters, plots, strand dumps) when you confirm saves at prompts. |

---

## Requirements

- **Python 3.x**
- **Core:** `numpy`, `matplotlib`
- **Multipass / lag-lead / integrated / variable:** `scipy`
- **kNN:** `scikit-learn`
- **CNN / bidir encoder / causal transformer:** `tensorflow`, `scikit-learn` (CNN legacy model loading)

Install a typical full stack:

```bash
pip install numpy matplotlib scipy scikit-learn tensorflow
```

For CPU-only TensorFlow, follow the [TensorFlow install guide](https://www.tensorflow.org/install) for your OS. You can run **method 1**, **single_image**, **multipass**, and **lag/lead/death** without TensorFlow; **CNN**, **bidir encoder**, and **causal transformer** need it.

---

## Implementation notes

- **`requirements.txt`** — `pip install -r requirements.txt`
- **Lag / lead / death fit** (`lagleaddeath.py`) uses a **fixed** forward model for each optimizer evaluation (no `random` in the loss). The fourth tuned parameter scales mild deterministic cross-talk in the same way on lag, lead, and survival terms.
- **Caller tweaks (population / geometry / ML):** `454Sim13` uses **Laplace-smoothed** counts for calls. **single_image** uses **L2-normalized** spots with a **tie band** → `N`. **kNN** uses **StandardScaler** + **distance weighting**; batched `predict()` for all windows per template. **CNN** predicts **per-position** (4 classes × window_size) instead of combo labels; `padding='same'` Conv layers; EarlyStopping; inputs ÷255. **Bidir Encoder** (`transformer8.py`) uses a **64-dim embedding**, **3 bidirectional encoder blocks** (4 heads × 16 key_dim), per-position softmax, EarlyStopping, and **auto-checkpoint warm-start** — it is NOT autoregressive (every position sees the full window). **Causal Transformer** (`causal_transformer.py`) is a **real GPT-style autoregressive model**: **4 causal blocks** (64-dim, 4 heads, 128-unit FFN) with **sinusoidal positional encoding** and `use_causal_mask=True` so position *t* only sees 0..*t*; processes the **full read** in one pass; auto-checkpoint warm-start across runs. **Multipass** uses **scipy.optimize (L-BFGS-B)** with ideal one-hot key colors, **joint estimation** across all templates for pass 1, a **noise_floor** parameter, **consensus voting** across overlapping windows, and **normalized (direction-based) error** for robustness at late cycles.
- **Honest accuracy:** ML methods (kNN, CNN, bidir encoder, causal transformer) report **test accuracy on unseen templates only** (those beyond `num_training_templates`), with training accuracy shown separately. Physics methods (single_image, multipass) report on all templates since they use no training data.
- **Performance:** All window-search callers (`multipass3`, `integrated2`, `variable`) use **precomputed combo arrays** and vectorized error via `np.einsum` (~50-100× faster than Python loops). ML callers use **batched inference** (one `predict()` call for all windows across all templates). Grid searches replaced with `scipy.optimize` or simple in-process loops (no `joblib` overhead).
- **454Sim13** sums **one blocked-terminator state per strand per cycle** (which dye sits on the end after that cycle). That matches **discrete SBS** with Lightning-style blocks: each successful step is its own block, **not** the classic 454 situation where **one flow** could add several identical bases and **light intensity** was used to guess homopolymer length. A run of **A·A·A** in the template still appears as **three cycles**, each with an A-channel signal—not as one ambiguous tall peak in a single cycle.

## License

If you publish this repository, add a `LICENSE` file with the terms you want (this README does not specify one).
