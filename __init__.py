"""chimneyml: multi-attribute machine-learning screening of gas chimneys in sparse 2D post-stack seismic data."""
from .conditioning import condition
from .attributes import compute, NAMES
from .classify import polygon_labels, blind_tests, trace_scores, pca_som
__version__ = '1.0.0'
