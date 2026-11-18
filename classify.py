"""Leave-one-line-out blind tests, permutation attribute ranking and PCA-SOM facies."""
import numpy as np
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from minisom import MiniSom


def polygon_labels(shape, polygons, background_trace_max=None, t_range_ms=(100, 1800), dt_ms=4.0, margin=10, exclude_traces=0):
    """Label array: 1 = chimney polygon, 0 = background, -1 = unused.
    polygons: list of [(trace, time_ms), ...]. Background = outside all polygons by > `margin` traces."""
    ns, nt = shape
    T, X = np.mgrid[0:ns, 0:nt]; tms = T * dt_ms; pts = np.c_[X.ravel(), tms.ravel()]
    ch = np.zeros(ns * nt, bool); near = np.zeros(ns * nt, bool)
    for poly in polygons:
        ch |= Path(poly).contains_points(pts)
        cx = np.array(poly, float)[:, 0].mean()
        near |= Path([(x + (margin if x > cx else -margin), t) for x, t in poly]).contains_points(pts)
    ch = ch.reshape(ns, nt); near = near.reshape(ns, nt)
    bg = (tms >= t_range_ms[0]) & (tms <= t_range_ms[1]) & ~near
    if background_trace_max is not None: bg &= X <= background_trace_max
    if exclude_traces: bg &= X > exclude_traces
    lab = np.full((ns, nt), -1, np.int8); lab[bg] = 0; lab[ch] = 1
    return lab


def draw(attrs, lab, n_pos, n_neg, rng):
    X = attrs.reshape(attrs.shape[0], -1).T; l = lab.ravel()
    ip = rng.choice(np.flatnonzero(l == 1), min(n_pos, (l == 1).sum()), replace=False)
    ineg = rng.choice(np.flatnonzero(l == 0), min(n_neg, (l == 0).sum()), replace=False)
    idx = np.r_[ip, ineg]; return X[idx], l[idx]


def make_models(seed=0):
    return {'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features='sqrt',
                                                    class_weight='balanced', n_jobs=-1, random_state=seed),
            'ANN (MLP)': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                                                       max_iter=400, early_stopping=True, random_state=seed))}


def blind_tests(attrs, labels, names, n_pos=3000, n_neg=9000, seed=42, smooth=(15, 1.5)):
    """attrs/labels: dicts line -> array. Returns metrics, RF probabilities, permutation importance (mean, sd)."""
    rng = np.random.default_rng(seed); lines = list(attrs)
    res, prob, imps = {}, {}, []
    for test in lines:
        tr = [n for n in lines if n != test]
        parts = [draw(attrs[n], labels[n], n_pos, n_neg, rng) for n in tr]
        Xtr = np.vstack([a for a, _ in parts]); ytr = np.concatenate([b for _, b in parts])
        Xall = attrs[test].reshape(attrs[test].shape[0], -1).T; l = labels[test].ravel(); m = l >= 0
        for mname, mdl in make_models().items():
            mdl.fit(Xtr, ytr); p = mdl.predict_proba(Xall)[:, 1]
            yt, pt = l[m], p[m]; yh = pt >= 0.5
            res[f'{mname}|{test}'] = dict(AUC=roc_auc_score(yt, pt), AP=average_precision_score(yt, pt), F1=f1_score(yt, yh),
                                          Precision=precision_score(yt, yh, zero_division=0), Recall=recall_score(yt, yh),
                                          BalancedAcc=balanced_accuracy_score(yt, yh))
            if mname == 'Random Forest':
                pm = p.reshape(attrs[test].shape[1:]); prob[test] = pm
                Xte, yte = draw(attrs[test], labels[test], 2000, 6000, rng)
                pi = permutation_importance(mdl, Xte, yte, scoring='roc_auc', n_repeats=8, random_state=0, n_jobs=-1)
                imps.append(pi.importances_mean)
    imp = np.array(imps)
    ranking = [(names[i], float(imp.mean(0)[i]), float(imp.std(0)[i])) for i in np.argsort(imp.mean(0))[::-1]]
    return res, prob, ranking


def trace_scores(prob, lab, t0_ms=150, t1_ms=1000, dt_ms=4.0, smooth=(15, 1.5)):
    """Trace-level score = mean smoothed RF probability in a time window; trace label = >50% chimney samples in window."""
    ps = gaussian_filter(prob, smooth)
    a, b = int(t0_ms / dt_ms), int(t1_ms / dt_ms)
    score = ps[a:b].mean(0); truth = (lab[a:b] == 1).mean(0) > 0.5
    return score, truth, (roc_auc_score(truth, score) if truth.any() and (~truth).any() else float('nan'))


def pca_som(attrs, labels, n_clusters=6, seed=0, n_train=60000, som_size=8, iters=20000, var=0.9, mask=None):
    """Unsupervised facies: PCA (90% variance) -> 8x8 SOM -> k-means on prototypes. Returns cluster maps and share table."""
    rng = np.random.default_rng(seed); lines = list(attrs)
    sl = {n: (slice(None), slice(None)) if mask is None else mask for n in lines}
    X = np.vstack([attrs[n].reshape(attrs[n].shape[0], -1).T for n in lines])
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=var, random_state=seed).fit(X[rng.choice(len(X), min(n_train, len(X)), replace=False)])
    Z = pca.transform(X); sub = Z[rng.choice(len(Z), min(n_train, len(Z)), replace=False)]
    som = MiniSom(som_size, som_size, Z.shape[1], sigma=1.5, learning_rate=0.5, random_seed=seed)
    som.pca_weights_init(sub); som.train_random(sub, iters)
    W = som.get_weights().reshape(-1, Z.shape[1]); km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(W)
    maps, off = {}, 0
    for n in lines:
        shp = attrs[n].shape[1:]; m = int(np.prod(shp)); Zn = Z[off:off + m]; off += m
        bmu = np.concatenate([np.argmin(((Zn[i:i + 20000, None, :] - W[None]) ** 2).sum(-1), 1) for i in range(0, m, 20000)])
        maps[n] = km.labels_[bmu].reshape(shp)
    cnt = np.zeros((n_clusters, 2))
    for n in lines:
        for k in range(n_clusters):
            cnt[k, 0] += ((maps[n] == k) & (labels[n] == 1)).sum(); cnt[k, 1] += ((maps[n] == k) & (labels[n] == 0)).sum()
    return maps, cnt / np.maximum(cnt.sum(0, keepdims=True), 1), pca.n_components_
