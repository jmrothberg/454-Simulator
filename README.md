# 454 Sequencing Simulator

Python simulations of **sequencing by synthesis** in the spirit of 454-style pyrosequencing / E‑wave chemistry with Lightning terminators: strand-level dynamics, UV and excitation timing, dark bases and strand loss, image-style color channels, and several **base callers** (including a small **transformer** model in TensorFlow).

Original research and code by **Jonathan M. Rothberg** (see file headers for dates and notes).

## Contents

| File | Role |
|------|------|
| `454Sim13.py` | Interactive strand simulator: prompts for timing parameters (`tauUV`, `uv_time`, `tauEX`, `ex_time`), `p_die` / `p_dark`, template DNA, and cycle count. Produces text visualization of strands and plots. |
| `MultiSim2.py` | **Multi-caller** pipeline: runs the simulator, builds per-cycle “images,” then lets you compare **KNN**, **transformer**, **multipass**, **CNN**, and lag/lead/death estimation (`lagleaddeath`). |
| `transformer8.py` | Transformer-based base calling; trains or loads models under a local `Sim Models/` directory (`.h5` filenames encode window size). |
| `cnn_caller.py` | 1D CNN base caller (TensorFlow + scikit-learn encoders). |
| `knn_caller4.py` | K-nearest-neighbors base caller. |
| `multipass3.py` | Multipass / window search base caller. |
| `lagleaddeath.py` | Estimates lag, lead, and related effects via optimization (`scipy`). |
| `integrated2.py` | Integrated iterative base calling with parallel grid search over lead/lag/death (`joblib`). |
| `variable.py` | Windowed base calling with uncertain lag/lead (exhaustive small-window search over base combinations). |

Sample run logs and parameters under `Sim Output/` are from local runs (optional to keep or ignore in version control).

## Requirements

- Python 3.x  
- `numpy`, `matplotlib`  
- `tensorflow` (for `transformer8.py` and `cnn_caller.py`)  
- `scikit-learn`, `scipy`, `joblib`

Install typical dependencies (versions depend on your OS/GPU):

```bash
pip install numpy matplotlib tensorflow scikit-learn scipy joblib
```

TensorFlow can be replaced with a CPU-only build if you do not need GPU acceleration; match the [TensorFlow install guide](https://www.tensorflow.org/install) to your platform.

## How to run

From this directory:

```bash
python 454Sim13.py
```

Follow the prompts for chemistry timing, error rates, template sequence, and number of cycles.

For the full multi-caller workflow:

```bash
python MultiSim2.py
```

The script imports the caller modules listed above; ensure you run it with the project root as the working directory so imports and `Sim Models/` resolve correctly.

## License

If you publish this repository, add a `LICENSE` file with the terms you want (this README does not specify one).
