import numpy as np

_min = np.minimum
_max = np.maximum

def _normalize(a):
    return a / np.linalg.norm(a)

def _vec(*arrs):
    return np.stack(arrs, axis=-1)

def _length(a):
    return np.linalg.norm(a, axis=1)

def _dot(a, b):
    return np.sum(a * b, axis=1)
