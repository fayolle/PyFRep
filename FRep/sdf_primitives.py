import numpy as np
from .utils import _min, _max, _length
from .utils import _normalize, _dot, _vec


def sphere(p, r):
    return r - _length(p)

def plane(p, normal=np.array((0, 0, 1)), point=np.array((0, 0, 0))):
    normal = _normalize(normal)
    return -np.dot(point - p, normal)

def box(p, b):
    q = np.abs(p) - b
    return -(_length(_max(q, 0)) + _min(np.amax(q, axis=1), 0))

def roundedBox(p, b, radius):
    q = np.abs(p) - b
    return -(_length(_max(q, 0)) + _min(np.amax(q, axis=1), 0) - radius)

def torus(p, r1, r2):
    xy = p[:,[0,1]]
    z = p[:,2]
    a = _length(xy) - r1
    return r2 - _length(_vec(a, z))

def capsule(p, a, b, radius):
    a = np.array(a)
    b = np.array(b)
    pa = p - a
    ba = b - a
    h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1).reshape((-1, 1))
    return radius - _length(pa - np.multiply(ba, h))

def cylinder(p, r):
    return r - _length(p[:,[0,1]])

def cappedCylinder(p, a, b, radius):
    a = np.array(a)
    b = np.array(b)
    ba = b - a
    pa = p - a
    baba = np.dot(ba, ba)
    paba = np.dot(pa, ba).reshape((-1, 1))
    x = _length(pa * baba - ba * paba) - radius * baba
    y = np.abs(paba - baba * 0.5) - baba * 0.5
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    x2 = x * x
    y2 = y * y * baba
    d = np.where(_max(x, y) < 0,-_min(x2, y2),np.where(x > 0, x2, 0) + np.where(y > 0, y2, 0))
    return -np.sign(d) * np.sqrt(np.abs(d)) / baba

def cappedCone(p, a, b, ra, rb):
    a = np.array(a)
    b = np.array(b)
    rba = rb - ra
    baba = np.dot(b - a, b - a)
    papa = _dot(p - a, p - a)
    paba = np.dot(p - a, b - a) / baba
    x = np.sqrt(papa - paba * paba * baba)
    cax = _max(0, x - np.where(paba < 0.5, ra, rb))
    cay = np.abs(paba - 0.5) - 0.5
    k = rba * rba + baba
    f = np.clip((rba * (x - ra) + paba * baba) / k, 0, 1)
    cbx = x - ra - f * rba
    cby = paba - f
    s = np.where(np.logical_and(cbx < 0, cay < 0), -1, 1)
    return -s * np.sqrt(_min(cax * cax + cay * cay * baba,cbx * cbx + cby * cby * baba))

