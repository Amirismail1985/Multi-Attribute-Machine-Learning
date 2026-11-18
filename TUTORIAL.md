# Tutorial: from synthetic lines to a blind-line detection map

```python
import json, numpy as np
import chimneyml as cm
from chimneyml.classify import polygon_labels, blind_tests, trace_scores

attrs, labels = {}, {}
for n in (1, 2, 3):
    z = np.load(f"data/synthetic/line_{n}.npz")
    d, polys = z["data"], json.loads(str(z["polygons"]))
    attrs[n] = cm.compute(d)                       # 15 x 575 x 300
    labels[n] = polygon_labels(d.shape, polys)     # 1 chimney, 0 background, -1 unused

res, prob, ranking = blind_tests(attrs, labels, cm.NAMES, n_pos=1500, n_neg=4500)
print(res["Random Forest|1"])                      # metrics on line 1, trained on lines 2 and 3
print(ranking[:3])                                 # most informative attributes
score, truth, auc = trace_scores(prob[1], labels[1])
print("trace-level ROC-AUC on the blind line:", auc)
```

1. **Run the generator** `python scripts/make_synthetic.py`. Each line has one or two vertical zones of dimmed, chaotic, low-passed reflections.
2. **Compute attributes.** `cm.compute` conditions the line and returns 15 laterally normalised attributes.
3. **Label.** Polygons give chimney samples; background is everything between 100 and 1800 ms that lies more than ten traces outside a polygon.
4. **Blind test.** `blind_tests` trains on two lines and predicts the third, for each line in turn.
5. **Read the outputs.** `res` holds ROC-AUC, average precision, F1, precision, recall and balanced accuracy per model and line; `prob[line]` is the probability map; `ranking` lists attributes by permutation importance.
6. **Plot** `prob[1]` with `matplotlib.pyplot.imshow(prob[1], aspect="auto", cmap="magma")` next to the amplitude section to see the chimney columns light up.
7. **Your own data.** Replace the loader with `d, dt = chimneyml.io.read_segy("line.sgy")`, draw polygons on your own sections, and keep at least three lines so that each blind test has two training lines.
