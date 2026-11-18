"""End-to-end demo on the synthetic lines: conditioning -> 15 attributes -> leave-one-line-out blind tests ->
permutation ranking -> PCA-SOM. Runs in a few minutes on a laptop CPU. Writes results to outputs/.

    python scripts/make_synthetic.py
    python scripts/run_demo.py
Replace the loader with `chimneyml.io.read_segy` to run on your own SEG-Y lines.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import chimneyml as cm
from chimneyml.classify import polygon_labels, blind_tests, trace_scores, pca_som

root = os.path.join(os.path.dirname(__file__), '..'); data = os.path.join(root, 'data', 'synthetic'); out = os.path.join(root, 'outputs')
os.makedirs(out, exist_ok=True)
t0 = time.time(); attrs, labels = {}, {}
for n in (1, 2, 3):
    z = np.load(os.path.join(data, f'line_{n}.npz')); d = z['data']; polys = json.loads(str(z['polygons']))
    attrs[f'L{n}'] = cm.compute(d)
    labels[f'L{n}'] = polygon_labels(d.shape, polys, background_trace_max=None, t_range_ms=(100, 1800))
    print(f'line {n}: attributes {attrs[f"L{n}"].shape}  ({time.time() - t0:.0f} s)')
res, prob, ranking = blind_tests(attrs, labels, cm.NAMES, n_pos=1500, n_neg=4500)
summary = {'blind_tests': {k: {m: round(float(v), 3) for m, v in r.items()} for k, r in res.items()}, 'ranking': [(a, round(m, 4), round(s, 4)) for a, m, s in ranking]}
for n in attrs:
    score, truth, auc = trace_scores(prob[n], labels[n]); summary[f'trace_auc_{n}'] = round(float(auc), 3)
    print(f'{n}: pixel RF AUC {res[f"Random Forest|{n}"]["AUC"]:.3f}   trace-level AUC {auc:.3f}')
print('top attributes:', [r[0] for r in ranking[:4]])
maps, share, npc = pca_som(attrs, labels, iters=5000, n_train=20000)
summary['pca_components_90pct'] = int(npc); summary['som_cluster_share_chimney_vs_background'] = np.round(share, 3).tolist()
json.dump(summary, open(os.path.join(out, 'demo_results.json'), 'w'), indent=1)
print(f'done in {time.time() - t0:.0f} s -> outputs/demo_results.json')
