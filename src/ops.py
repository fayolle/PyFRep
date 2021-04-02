import numpy as np
from utils import _vec


def scale(p, factor):
    try:
        x, y, z = factor
    except TypeError:
        x = y = z = factor
    s = (x, y, z)
    return (p / s)

def twist(p, k):
    x = p[:,0]
    y = p[:,1]
    z = p[:,2]
    c = np.cos(k * z)
    s = np.sin(k * z)
    x2 = c * x - s * y
    y2 = s * x + c * y
    z2 = z
    return _vec(x2, y2, z2)

def bend(p, k):
    x = p[:,0]
    y = p[:,1]
    z = p[:,2]
    c = np.cos(k * x)
    s = np.sin(k * x)
    x2 = c * x - s * y
    y2 = s * x + c * y
    z2 = z
    return _vec(x2, y2, z2)

def union(d1, d2):
    d = d1 + d2 + np.sqrt(d1**2 + d2**2)
    return d

def difference(d1, d2):
    d = d1 - d2 - np.sqrt(d1**2 + d2**2)
    return d

def intersection(d1, d2):
    d = d1 + d2 - np.sqrt(d1**2 + d2**2)
    return d

def negate(d1):
    return -d1
