# User guide

## Conventions
- A line is a `float32` array `d` with shape `(n_samples, n_traces)`; the sample interval is 4 ms (`chimneyml.conditioning.DT`). Change `DT` before importing the package if your data use another interval; the filter lengths in samples (for example the 11-sample RMS window) are tuned for 4 ms.
- Polygons are lists of `(trace_index, time_ms)` vertices.
- Labels are `int8` arrays: `1` chimney, `0` background, `-1` unused.

## Functions
| Function | Purpose |
|---|---|
| `chimneyml.condition(d, f1=6, f2=60)` | Returns `(band-passed, conditioned)` data. |
| `chimneyml.compute(d)` | Returns the attribute cube `(15, n_samples, n_traces)`; `chimneyml.NAMES` lists the attributes. |
| `polygon_labels(shape, polygons, background_trace_max=None, t_range_ms=(100, 1800), margin=10, exclude_traces=0)` | Builds labels. Background excludes samples within `margin` traces of a polygon. |
| `blind_tests(attrs, labels, names, n_pos, n_neg)` | `attrs` and `labels` are dicts keyed by line name. Returns metrics per model and test line, Random Forest probability maps, and the permutation ranking (name, mean AUC drop, SD across folds). |
| `trace_scores(prob, lab, t0_ms=150, t1_ms=1000)` | Gaussian-smooths the probability map and returns trace scores, trace truth and trace-level ROC-AUC. |
| `pca_som(attrs, labels, n_clusters=6)` | Returns cluster maps, the share of chimney and background samples per cluster, and the number of principal components. |

## Parameters used in the paper
Conditioning: spike clip 18 x median absolute amplitude, 6-60 Hz fourth-order zero-phase Butterworth, 0.5 s balancing windows, median over 7 traces along dip, linearity gate 0.3-0.8, two iterations. Random Forest: 300 trees, minimum leaf 5, sqrt(15) attributes per split, balanced class weights. ANN: 32-16 hidden units, L2 alpha 1e-3, early stopping. Training draw: 3000 chimney and 9000 background samples per training line. SOM: 8 x 8, sigma 1.5, learning rate 0.5, 20,000 iterations, six k-means clusters.

## Limits
- Labels must be drawn by the user; the classifier cannot separate chimneys from faults unless both are labelled as separate classes (not implemented).
- The lateral normalisation uses the distribution of the line being predicted, so normalised attributes are anomalies relative to the same time level elsewhere on the line, not rock properties.
- Leave-one-line-out blind tests evaluate transfer within one survey, not generalisation to independent chimney systems.

## Reproducibility
Seeds are fixed (`seed=42` for sampling, `random_state=0` for models). Results can still differ slightly across library versions and CPU counts.
