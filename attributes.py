"""Fifteen seismic attributes, each normalised laterally (robust z-score along the line, per time sample).
Input: 2D post-stack line, shape (n_samples, n_traces), sample interval DT seconds."""
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from skimage.filters.rank import entropy
try:
    from skimage.morphology import footprint_rectangle as _rect
except ImportError:  # older scikit-image
    from skimage.morphology import rectangle as _rect
from .conditioning import bandpass, structure_tensor, semblance, condition, DT

NAMES = ['Envelope', 'RMS amplitude', 'Sweetness', 'Instantaneous frequency', 'Cosine of phase',
         'Relative impedance', 'Spec. decomp. 15 Hz', 'Spec. decomp. 30 Hz', 'Spec. decomp. 45 Hz',
         'HF/LF ratio', 'Variance', 'Dip magnitude', 'Reflector linearity', 'Lateral time-shift',
         'GLCM-type entropy']


def lateral_norm(a, clip=6.0):
    med = np.median(a, 1, keepdims=True)
    mad = np.median(np.abs(a - med), 1, keepdims=True) * 1.4826 + 1e-9
    return np.clip((a - med) / mad, -clip, clip).astype(np.float32)


def lag_residual(d, h=3, win=31, maxlag=6):
    ns, nt = d.shape
    L = np.zeros((ns, nt), np.float32); best = np.full((ns, nt), -9., np.float32)
    for l in range(-maxlag, maxlag + 1):
        b = np.roll(d, -l, axis=0)
        a0 = d[:, :-2 * h]; b0 = b[:, 2 * h:]
        num = uniform_filter(a0 * b0, (win, 1))
        den = np.sqrt(uniform_filter(a0 * a0, (win, 1)) * uniform_filter(b0 * b0, (win, 1))) + 1e-9
        cc = num / den; m = cc > best[:, h:-h]
        best[:, h:-h][m] = cc[m]; L[:, h:-h][m] = l
    return np.abs(L - median_filter(L, (1, 31)))


def compute(d):
    """Return the attribute cube A with shape (15, n_samples, n_traces)."""
    d = np.asarray(d, np.float32)
    bp, cond = condition(d)
    an = hilbert(bp, axis=0); env = np.abs(an); ph = np.unwrap(np.angle(an), axis=0)
    ifreq = np.clip(gaussian_filter(np.gradient(ph, axis=0) / (2 * np.pi * DT), (5, 1)), 1, 80)
    envs = gaussian_filter(env, (3, 1))
    rms = np.sqrt(uniform_filter(bp ** 2, (11, 1)))
    sweet = envs / np.sqrt(ifreq)
    cosph = np.cos(np.angle(an))
    rai = np.cumsum(bp, axis=0); rai = rai - gaussian_filter(rai, (25, 0))
    sd = [gaussian_filter(np.abs(hilbert(bandpass(d, f - 5, f + 5), axis=0)), (5, 1)) for f in (15, 30, 45)]
    lo = gaussian_filter(np.abs(hilbert(bandpass(d, 8, 20), axis=0)), (8, 2))
    hi = gaussian_filter(np.abs(hilbert(bandpass(d, 35, 55), axis=0)), (8, 2))
    ratio = np.log10(hi / (lo + 1e-6))
    var = gaussian_filter(1 - semblance(cond), (3, 1))
    p, lin = structure_tensor(cond)
    dip = np.abs(p); lin = gaussian_filter(lin, (3, 1))
    lag = gaussian_filter(lag_residual(bp), (4, 1))
    q = np.clip((cond / (np.percentile(np.abs(cond), 99) + 1e-9) + 1) * 127.5, 0, 255).astype(np.uint8)
    ent = entropy(q, _rect((9, 5)) if _rect.__name__ == 'footprint_rectangle' else _rect(9, 5)).astype(np.float32)
    raw = [envs, rms, sweet, ifreq, cosph, rai, *sd, ratio, var, dip, lin, lag, ent]
    return np.stack([lateral_norm(a) for a in raw], 0)
