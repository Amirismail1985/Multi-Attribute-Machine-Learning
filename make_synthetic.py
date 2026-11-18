"""Generate three synthetic parallel 2D post-stack lines with embedded 'chimneys' (open test data, CC0).

Each line is (575 samples x 300 traces), dt = 4 ms, 25 m trace spacing. Layered, gently folded reflectivity is
convolved with a 30 Hz Ricker wavelet. Inside a chimney the reflectivity is dimmed and made chaotic, and the
trace is low-passed (loss of reflector continuity, amplitude and high frequencies). Noise is added.
Output: data/synthetic/line_{1,2,3}.npz with arrays `data` and `polygons` (trace, time_ms vertices).
"""
import numpy as np, os, json
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter1d

NS, NT, DT = 575, 300, 0.004
CHIMNEYS = {1: [(150, 175, 1200)], 2: [(140, 168, 1500), (230, 245, 900)], 3: [(145, 170, 1400)]}


def ricker(f=30.0, n=41):
    t = (np.arange(n) - n // 2) * DT; a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def make_line(seed, chimneys):
    rng = np.random.default_rng(seed)
    refl = np.zeros((NS, NT), np.float32)
    x = np.arange(NT)
    fold = 6 * np.sin(2 * np.pi * x / 220 + seed) + 0.01 * x
    for k in range(70):
        t0 = rng.uniform(60, NS - 40); amp = rng.normal(0, 1) * (1 + 0.8 * (t0 < 200))
        tt = np.clip(np.round(t0 + fold * (0.4 + t0 / NS)).astype(int), 0, NS - 1)
        refl[tt, x] += amp
    polys = []
    for (a, b, base) in chimneys:
        top = 30; base_s = int(base / (DT * 1000))
        wob = 1.5 * np.sin(np.linspace(0, 6, base_s - top))
        for i, s in enumerate(range(top, base_s)):
            lo, hi = int(a + wob[i]), int(b + wob[i])
            refl[s, lo:hi] = refl[s, lo:hi] * 0.2 + rng.normal(0, 0.7, hi - lo)
        polys.append([(a, top * 4), (a, base), (b, base), (b, top * 4)])
    d = fftconvolve(refl, ricker()[:, None], mode='same', axes=0)
    for (a, b, base) in chimneys:
        sl = (slice(30, int(base / 4)), slice(a, b))
        d[sl] = gaussian_filter1d(d[sl], 2.5, axis=0)
    d += rng.normal(0, 0.15 * d.std(), d.shape)
    return d.astype(np.float32), polys


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic'); os.makedirs(out, exist_ok=True)
    for n, ch in CHIMNEYS.items():
        d, polys = make_line(n, ch)
        np.savez_compressed(os.path.join(out, f'line_{n}.npz'), data=d, polygons=json.dumps(polys))
        print('wrote line', n, d.shape)
