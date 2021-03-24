import numpy as np


_min = np.minimum
_max = np.maximum


def _normalize(a):
    return a / np.linalg.norm(a)

def _vec(*arrs):
    return np.stack(arrs, axis=-1)

def translate(p, offset):
    return p-offset

def scale(p, factor):
    try:
        x, y, z = factor
    except TypeError:
        x = y = z = factor
    s = (x, y, z)
    m = min(x, min(y, z))

    # Note: m should be used to correct the SDF value
    # i.e. m * f(p/s)
    return (p / s)

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
