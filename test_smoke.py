import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import chimneyml as cm
from chimneyml.classify import polygon_labels, blind_tests, trace_scores
from make_synthetic import make_line, CHIMNEYS


def test_attribute_cube_shape_and_finite():
    d, _ = make_line(1, CHIMNEYS[1])
    A = cm.compute(d)
    assert A.shape == (15, *d.shape) and np.isfinite(A).all()
    assert np.abs(A).max() <= 6.0 + 1e-6          # robust z-score is clipped at +/-6


def test_blind_line_detects_synthetic_chimney():
    attrs, labels = {}, {}
    for n in (1, 2, 3):
        d, polys = make_line(n, CHIMNEYS[n]); attrs[n] = cm.compute(d); labels[n] = polygon_labels(d.shape, polys)
    res, prob, ranking = blind_tests(attrs, labels, cm.NAMES, n_pos=800, n_neg=2400)
    for n in attrs:
        assert res[f'Random Forest|{n}']['AUC'] > 0.9
        _, _, auc = trace_scores(prob[n], labels[n]); assert auc > 0.9
    assert len(ranking) == 15
