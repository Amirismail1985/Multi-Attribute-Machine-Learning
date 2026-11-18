# Multi-Attribute Machine Learning for Gas-Chimney Screening (`chimneyml`)

Reproducible multi-attribute machine-learning workflow for screening **gas chimneys** in sparse **2D post-stack seismic** data, with **leave-one-line-out blind validation**.
Companion code for: *A Reproducible Multi-Attribute Machine-Learning Workflow with Blind-Line Validation for Gas-Chimney Screening in 2D Seismic Data: Application to the Shallow-Water Eastern Nile Delta, Egypt* (submitted to Mathematical Geosciences; Online Resource 1).

## What it does
1. **Conditioning** (`chimneyml.conditioning`): despike, zero-phase band-pass, time-variant spectral balancing, and an edge-preserving dip-steered median filter gated by reflector linearity.
2. **Fifteen attributes** (`chimneyml.attributes`): complex-trace, spectral-decomposition, geometric and texture attributes, each normalised laterally with a robust z-score (removes depth trends).
3. **Blind tests** (`chimneyml.classify.blind_tests`): Random Forest and neural network trained on all lines except one, scored on the withheld line; permutation attribute ranking; trace-level scoring.
4. **Unsupervised check** (`chimneyml.classify.pca_som`): PCA (90 % variance) followed by an 8 x 8 SOM and k-means clusters.

## Install
Python >= 3.10.
```bash
git clone https://github.com/Amirismail1985/Multi-Attribute-Machine-Learning && cd Multi-Attribute-Machine-Learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start (synthetic data, about 1 minute)
```bash
python scripts/make_synthetic.py   # writes data/synthetic/line_{1,2,3}.npz
python scripts/run_demo.py         # writes outputs/demo_results.json
pytest -q                          # smoke tests
```
The synthetic lines are an open test set (CC0) for checking the installation and following the tutorial. Their chimneys are deliberately easy to detect, so the scores they give say nothing about performance on field data.

## Use on your own data
See `docs/USER_GUIDE.md` and `docs/TUTORIAL.md`. In short: read each line to an array of shape `(n_samples, n_traces)` (`chimneyml.io.read_segy` for SEG-Y), call `chimneyml.compute(d)`, draw chimney polygons as `(trace, time_ms)` vertices, build labels with `polygon_labels`, and run `blind_tests`.

## Data
The field data of the paper (Port Fouad Marine, eastern offshore Nile Delta) belong to the Egyptian General Petroleum Corporation (EGPC) and Rashpetco and cannot be redistributed.

## Layout
```
chimneyml/    package (conditioning, attributes, classify, io)
scripts/      make_synthetic.py, run_demo.py
data/         synthetic test lines (generated)
docs/         USER_GUIDE.md, TUTORIAL.md
tests/        pytest smoke tests
```

## Licence
MIT (see `LICENSE`). Cite the article if you use the code (`CITATION.cff`).
