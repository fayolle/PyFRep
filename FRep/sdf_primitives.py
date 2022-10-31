import torch
import math
from .utils import _min, _max, _length
from .utils import _normalize, _dot, _vec


def sphere(p, center, r):
    if not torch.is_tensor(center):
        center = torch.tensor(center)
    return r - _length(p - center)


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


def torus(p, r1, r2):
    xy = p[:, [0, 1]]
    z = p[:, 2]
    a = _length(xy) - r1
    return r2 - _length(_vec(a, z))


def capsule(p, a, b, radius):
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    if not torch.is_tensor(b):
        b = torch.tensor(b)
    pa = p - a
    ba = b - a
    h = torch.clip(torch.dot(pa, ba) / torch.dot(ba, ba), 0.0, 1.0).reshape(
        (-1, 1))
    return radius - _length(pa - torch.multiply(ba, h))


def cylinderX(p, center, r):
    if not torch.is_tensor(center):
        center = torch.tensor(center)
    return r - _length(p[:, [1, 2]] - center[[1, 2]])


def cylinderY(p, center, r):
    if not torch.is_tensor(center):
        center = torch.tensor(center)
    return r - _length(p[:, [0, 2]] - center[[0, 2]])


def cylinderZ(p, center, r):
    if not torch.is_tensor(center):
        center = torch.tensor(center)
    return r - _length(p[:, [0, 1]] - center[[0, 1]])


def cappedCylinder(p, a, b, radius):
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    if not torch.is_tensor(b):
        b = torch.tensor(b)

    ba = b - a
    pa = p - a

    baba = torch.dot(ba, ba)
    #paba = torch.dot(pa, ba).reshape((-1, 1))
    paba = torch.matmul(pa, ba)
    paba = paba.reshape((-1,1))

    x = _length(pa * baba - ba * paba) - radius * baba
    y = torch.abs(paba - baba * 0.5) - baba * 0.5
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    x2 = x * x
    y2 = y * y * baba

    xtmp = torch.where(x>0, x2, torch.Tensor([0.]))
    ytmp = torch.where(y>0, y2, torch.Tensor([0.]))

    '''
    d = torch.where(
        _max(x, y) < 0, -_min(x2, y2),
        torch.where(x > 0, x2, 0.0) + torch.where(y > 0, y2, 0.0))
    '''
    d = torch.where(_max(x,y)<0, -_min(x2,y2), xtmp+ytmp)

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
    return -s * torch.sqrt(
        _min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba))


# Primitives from the HF library
def block(x, vertex, dx, dy, dz):
    if not torch.is_tensor(vertex):
        vertex = torch.tensor(vertex)
    b = torch.tensor((dx, dy, dz))
    shift = vertex + 0.5 * b
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
    xt = x[:, 0] - center[0]
    yt = x[:, 1] - center[1]
    zt = x[:, 2] - center[2]
    dist = torch.sqrt(zt * zt + yt * yt) * ct - torch.abs(xt) * st
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
    xt = x[:, 0] - center[0]
    yt = x[:, 1] - center[1]
    zt = x[:, 2] - center[2]
    dist = torch.sqrt(zt * zt + xt * xt) * ct - torch.abs(yt) * st
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
    xt = x[:, 0] - center[0]
    yt = x[:, 1] - center[1]
    zt = x[:, 2] - center[2]
    dist = torch.sqrt(xt * xt + yt * yt) * ct - torch.abs(zt) * st
    return -dist


# General cylinder
# Pass through point 'center' in direction 'u' with radius 'r'
def cylinder(x, center, u, r):
    if not torch.is_tensor(center):
        center = torch.tensor(center)
    if not torch.is_tensor(u):
        u = torch.tensor(u)

    cu = center + u
    cmx = center - x

    # broadcast for the cross-product
    ub = torch.zeros_like(cmx)
    ub[:] = u
    cp = torch.cross(ub, cmx)

    d1 = cp[:, 0]**2 + cp[:, 1]**2 + cp[:, 2]**2
    d2 = u[0]**2 + u[1]**2 + u[2]**2

    d = d1 / d2
    d = torch.sqrt(d)

    f = r - d
    return f


def torusX(x, center, R, r0):
    if not torch.is_tensor(center):
        center = torch.tensor(center)

    x2 = x - center
    dyz = torch.sqrt(x2[:, 1]**2 + x2[:, 2]**2)
    dyz = dyz - R
    dyzx = torch.sqrt(dyz**2 + x2[:, 0]**2)
    dyzx = dyzx - r0
    return -dyzx


def torusY(x, center, R, r0):
    if not torch.is_tensor(center):
        center = torch.tensor(center)

    x2 = x - center
    dxz = torch.sqrt(x2[:, 0]**2 + x2[:, 2]**2)
    dxz = dxz - R
    dxzy = torch.sqrt(dxz**2 + x2[:, 1]**2)
    dxzy = dxzy - r0
    return -dxzy


def torusZ(x, center, R, r0):
    if not torch.is_tensor(center):
        center = torch.tensor(center)

    x2 = x - center
    dxy = torch.sqrt(x2[:, 0]**2 + x2[:, 1]**2)
    dxy = dxy - R
    dxyz = torch.sqrt(dxy**2 + x2[:, 2]**2)
    dxyz = dxyz - r0
    return -dxyz


def plane(p, n, h):
    if not torch.is_tensor(n):
        n = torch.tensor(n)

    return -(p[:,0]*n[:,0] + p[:,1]*n[:,1] + p[:,2]*n[:,2] + h)


def frame(p, side_length, thickness):
    if not torch.is_tensor(side_length):
        side_length = torch.tensor(side_length)
    if not torch.is_tensor(thickness):
        thickness = torch.tensor(thickness)

    pt = torch.abs(p) - side_length/2.0 - thickness/2.0
    qt = torch.abs(pt + thickness/2.0) - thickness/2.0
    ptx, pty, ptz = pt[:,0], pt[:,1], pt[:,2]
    qtx, qty, qtz = qt[:,0], qt[:,1], qt[:,2]

    z1 = torch.zeros_like(pt)
    z2 = torch.zeros_like(ptx)

    t1 = _length(_max(_vec(ptx, qty, qtz), z1)) + _min(_max(ptx, _max(qty, qtz)), z2)
    t2 = _length(_max(_vec(qtx, pty, qtz), z1)) + _min(_max(qtx, _max(pty, qtz)), z2)
    t3 = _length(_max(_vec(qtx, qty, ptz), z1)) + _min(_max(qtx, _max(qty, ptz)), z2)

    return -(_min(_min(t1, t2), t3))
