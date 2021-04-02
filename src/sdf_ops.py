import numpy as np
from utils import _normalize, _vec, _min, _max


def translate(p, offset):
    return p-offset

def rotate(p, angle, vector=np.array((0, 0, 1))):
    x, y, z = _normalize(vector)
    s = np.sin(angle)
    c = np.cos(angle)
    m = 1 - c
    matrix = np.array([
        [m*x*x + c, m*x*y + z*s, m*z*x - y*s],
        [m*x*y - z*s, m*y*y + c, m*y*z + x*s],
        [m*z*x + y*s, m*y*z - x*s, m*z*z + c],
    ]).T
    return np.dot(p, matrix)

# Transform such that (0,0,1) becomes v
def orient(p, v):
    v1 = np.array((0,0,1))
    v = _normalize(v)
    d = np.dot(v1, v)
    a = np.arccos(d)
    v2 = np.cross(v, v1)
    return rotate(p, a, v2)

def rep(p, c):
    q = mod(p + 0.5*c, c)-0.5*c
    return q

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
