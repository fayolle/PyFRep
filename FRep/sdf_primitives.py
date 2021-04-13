import torch
import math
from .utils import _min, _max, _length
from .utils import _normalize, _dot, _vec


def sphere(p, r):
    return r - _length(p)

def plane(p, normal=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0)):
    if not torch.is_tensor(normal):
        normal = torch.tensor(normal)
    if not torch.is_tensor(point):
        point = torch.tensor(point)
    
    normal = _normalize(normal)
    return -torch.dot(point - p, normal)

def box(p, b):
    if not torch.is_tensor(b):
        b = torch.tensor(b)
    q = torch.abs(p) - b
    z = torch.zeros_like(q)
    t1 = _length(_max(q, z))

    t21 = torch.amax(q, axis=1)
    t22 = torch.zeros_like(t21)
    
    t2 = _min(t21, t22)
    
    return -(t1 + t2)
    #return -(_length(_max(q, z)) + _min(torch.amax(q, axis=1), z))

def roundedBox(p, b, radius):
    if not torch.is_tensor(b):
        b = torch.tensor(b)
    q = torch.abs(p) - b
    z = torch.zeros_like(q)
    t1 = _length(_max(q, z))

    t21 = torch.amax(q, axis=1)
    t22 = torch.zeros_like(t21)
    t2 = _min(t21, t22)

    return -(t1 + t2 - radius)
    #return -(_length(_max(q, z)) + _min(torch.amax(q, axis=1), z) - radius)

def torus(p, r1, r2):
    xy = p[:,[0,1]]
    z = p[:,2]
    a = _length(xy) - r1
    return r2 - _length(_vec(a, z))

def capsule(p, a, b, radius):
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    if not torch.is_tensor(b):
        b = torch.tensor(b)
    pa = p - a
    ba = b - a
    h = torch.clip(torch.dot(pa, ba) / torch.dot(ba, ba), 0.0, 1.0).reshape((-1, 1))
    return radius - _length(pa - torch.multiply(ba, h))

def cylinder(p, r):
    return r - _length(p[:,[0,1]])

def cappedCylinder(p, a, b, radius):
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    if not torch.is_tensor(b):
        b = torch.tensor(b)
    ba = b - a
    pa = p - a
    baba = torch.dot(ba, ba)
    paba = torch.dot(pa, ba).reshape((-1, 1))
    x = _length(pa * baba - ba * paba) - radius * baba
    y = torch.abs(paba - baba * 0.5) - baba * 0.5
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    x2 = x * x
    y2 = y * y * baba
    d = torch.where(_max(x, y) < 0,-_min(x2, y2),torch.where(x > 0, x2, 0) + torch.where(y > 0, y2, 0))
    return -torch.sign(d) * torch.sqrt(torch.abs(d)) / baba

def cappedCone(p, a, b, ra, rb):
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    if not torch.is_tensor(b):
        b = torch.tensor(b)
    rba = rb - ra
    baba = torch.dot(b - a, b - a)
    papa = _dot(p - a, p - a)
    paba = torch.dot(p - a, b - a) / baba
    x = torch.sqrt(papa - paba * paba * baba)
    cax = _max(0, x - torch.where(paba < 0.5, ra, rb))
    cay = torch.abs(paba - 0.5) - 0.5
    k = rba * rba + baba
    f = torch.clip((rba * (x - ra) + paba * baba) / k, 0, 1)
    cbx = x - ra - f * rba
    cby = paba - f
    s = torch.where(torch.logical_and(cbx < 0, cay < 0), -1, 1)
    return -s * torch.sqrt(_min(cax * cax + cay * cay * baba,cbx * cbx + cby * cby * baba))

# Primitives from the HF library
def block(x, vertex, dx, dy, dz):
    if not torch.is_tensor(vertex):
        vertex = torch.tensor(vertex)
    b = torch.tensor((dx,dy,dz))
    shift = vertex + 0.5*b
    xt = x - shift
    return box(xt, b)

def coneX(x, center, r):
    if torch.is_tensor(r):
        t = torch.atan(r)
        ct = torch.cos(t)
        st = torch.sin(t)
    else:
        t = math.atan(r)
        ct = math.cos(t)
        st = math.sin(t)
    xt = x[:,0] - center[0]
    yt = x[:,1] - center[1]
    zt = x[:,2] - center[2]
    dist = torch.sqrt(zt*zt+yt*yt)*ct - torch.abs(xt)*st
    return -dist

def coneY(x, center, r):
    if torch.is_tensor(r):
        t = torch.atan(r)
        ct = torch.cos(t)
        st = torch.sin(t)
    else:
        t = math.atan(r)
        ct = math.cos(t)
        st = math.sin(t)
    xt = x[:,0] - center[0]
    yt = x[:,1] - center[1]
    zt = x[:,2] - center[2]
    dist = torch.sqrt(zt*zt+xt*xt)*ct - torch.abs(yt)*st
    return -dist

def coneZ(x, center, r):
    if torch.is_tensor(r):
        t = torch.atan(r)
        ct = torch.cos(t)
        st = torch.sin(t)
    else:
        t = math.atan(r)
        ct = math.cos(t)
        st = math.sin(t)
    xt = x[:,0] - center[0]
    yt = x[:,1] - center[1]
    zt = x[:,2] - center[2]
    dist = torch.sqrt(xt*xt+yt*yt)*ct - torch.abs(zt)*st
    return -dist
