import numpy as np
from .utils import _min, _max


def union(d1, d2):
    d = _max(d1, d2)
    return d

def difference(d1, d2):
    d = _min(d1, -d2)
    return d

def intersection(d1, d2):
    d = _min(d1, d2)
    return d

def negate(d1):
    return -d1

def dilate(d1, r):
    return d1 + r

def erode(d1, r):
    return d1 - r

def shell(d1, e):
    return -np.abs(d1) + e
