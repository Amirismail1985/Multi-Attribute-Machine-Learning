"""Optional SEG-Y reader (requires `segyio`). Returns an array of shape (n_samples, n_traces)."""
import numpy as np


def read_segy(path):
    import segyio
    with segyio.open(path, ignore_geometry=True) as f:
        d = segyio.tools.collect(f.trace[:]).T.astype(np.float32)
        dt = segyio.tools.dt(f) / 1e6
    return d, dt
