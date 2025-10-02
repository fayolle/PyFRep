import torch
import math
from .utils import _normalize, _vec, _min, _max


def scale(p, factor):
    try:
        x, y, z = factor
    except TypeError:
        x = y = z = factor
    s = torch.tensor((x, y, z))
    return (p / s)


def twist(p, k):
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    c = torch.cos(k * z)
    s = torch.sin(k * z)
    x2 = c * x - s * y
    y2 = s * x + c * y
    z2 = z
    return _vec(x2, y2, z2)


def bend(p, k):
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    c = torch.cos(k * x)
    s = torch.sin(k * x)
    x2 = c * x - s * y
    y2 = s * x + c * y
    z2 = z
    return _vec(x2, y2, z2)


def union(d1, d2):
    d = d1 + d2 + torch.sqrt(d1**2 + d2**2)
    return d


def union_m(d1, d2, m=2):
    nd = torch.sqrt(d1**2 + d2**2)
    d = (d1 + d2 + nd)*(nd**m)
    return d


def difference(d1, d2):
    d = d1 - d2 - torch.sqrt(d1**2 + d2**2)
    return d


def difference_m(d1, d2, m=2):
    nd = torch.sqrt(d1**2 + d2**2)
    d = (d1 - d2 - nd)*(nd**m)
    return d


def intersection(d1, d2):
    d = d1 + d2 - torch.sqrt(d1**2 + d2**2)
    return d


def intersection_m(d1, d2, m=2):
    nd = torch.sqrt(d1**2 + d2**2)
    d = (d1 + d2 - nd)*(nd**m)
    return d


def negate(d1):
    return -d1


def scale3D(p, sx, sy, sz):
    s = torch.tensor((sx, sy, sz))
    return (p / s)


def shift3D(p, dx, dy, dz):
    return p - torch.tensor((dx, dy, dz))


def translate(p, offset):
    return p - offset


def rotate3DX(p, theta):
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta)
    p2 = p.clone()
    p2[:, 1] = p[:, 1] * torch.cos(theta) + p[:, 2] * torch.sin(theta)
    p2[:, 2] = -p[:, 1] * torch.sin(theta) + p[:, 2] * torch.cos(theta)
    return p2


def rotate3DY(p, theta):
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta)
    p2 = p.clone()
    p2[:, 2] = p[:, 2] * torch.cos(theta) + p[:, 0] * torch.sin(theta)
    p2[:, 0] = -p[:, 2] * torch.sin(theta) + p[:, 0] * torch.cos(theta)
    return p2


def rotate3DZ(p, theta):
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta)
    p2 = p.clone()
    p2[:, 0] = p[:, 0] * torch.cos(theta) + p[:, 1] * torch.sin(theta)
    p2[:, 1] = -p[:, 0] * torch.sin(theta) + p[:, 1] * torch.cos(theta)
    return p2


def rotate(p, angle, vector=(0.0, 0.0, 1.0)):
    if (not torch.is_tensor(vector)):
        vector = torch.tensor(vector)
    x, y, z = _normalize(vector)
    s = torch.sin(angle)
    c = torch.cos(angle)
    m = 1 - c
    matrix = torch.tensor([
        [m * x * x + c, m * x * y + z * s, m * z * x - y * s],
        [m * x * y - z * s, m * y * y + c, m * y * z + x * s],
        [m * z * x + y * s, m * y * z - x * s, m * z * z + c],
    ])
    #torch.t(matrix)
    matrix = matrix.T
    return torch.matmul(p, matrix)
    #return torch.dot(p, matrix)


# Transform such that (0,0,1) becomes v
def orient(p, v):
    v = torch.tensor(v)
    v1 = torch.tensor((0.0, 0.0, 1.0))
    v = _normalize(v)
    d = torch.dot(v1, v)
    a = torch.arccos(d)
    v2 = torch.cross(v, v1)
    return rotate(p, a, v2)


def blendUnion(f1, f2, a0, a1, a2):
    t = f1 + f2 + torch.sqrt(f1**2 + f2**2)
    f1a1 = f1 / a1
    f2a2 = f2 / a2
    disp = a0 / (1.0 + f1a1**2 + f2a2**2)
    return t + disp


def blendIntersection(f1, f2, a0, a1, a2):
    t = f1 + f2 - torch.sqrt(f1**2 + f2**2)
    f1a1 = f1 / a1
    f2a2 = f2 / a2
    disp = a0 / (1.0 + f1a1**2 + f2a2**2)
    return t + disp


def twistX(p, x1, x2, theta1, theta2):
    p2 = p.clone()
    t = (p[:, 0] - x1) / (x2 - x1)
    theta = (1.0 - t) * theta1 + t * theta2
    p2[:, 1] = p[:, 1] * torch.cos(theta) + p[:, 2] * torch.sin(theta)
    p2[:, 2] = -p[:, 1] * torch.sin(theta) + p[:, 2] * torch.cos(theta)
    return p2


def twistY(p, y1, y2, theta1, theta2):
    p2 = p.clone()
    t = (p[:, 1] - y1) / (y2 - y1)
    theta = (1.0 - t) * theta1 + t * theta2
    p2[:, 2] = p[:, 2] * torch.cos(theta) + p[:, 0] * torch.sin(theta)
    p2[:, 0] = -p[:, 2] * torch.sin(theta) + p[:, 0] * torch.cos(theta)
    return p2


def twistZ(p, z1, z2, theta1, theta2):
    p2 = p.clone()
    t = (p[:, 1] - z1) / (z2 - z1)
    theta = (1.0 - t) * theta1 + t * theta2
    p2[:, 0] = p[:, 0] * torch.cos(theta) + p[:, 1] * torch.sin(theta)
    p2[:, 1] = -p[:, 0] * torch.sin(theta) + p[:, 1] * torch.cos(theta)
    return p2


def stretch3D(p, x0, sx, sy, sz):
    p2 = p.clone()
    p2[:, 0] = x0[0] + (p[:, 0] - x0[0]) / sx
    p2[:, 1] = x0[1] + (p[:, 1] - x0[1]) / sy
    p2[:, 2] = x0[2] + (p[:, 2] - x0[2]) / sz
    return p2


def taperX(p, x1, x2, s1, s2):
    p2 = p.clone()
    #scale = torch.zeros((p.shape[0], 1))
    scale = torch.zeros(p.shape[0])
    scale[:] = s1
    idx = (p[:, 0] > x2)
    scale[idx] = s2
    idx = (p[:, 0] >= x1) & (p[:, 0] <= x2)
    t = (p[:, 0] - x1) / (x2 - x1)
    scale[idx] = (1.0 - t[idx]) * s1 + t[idx] * s2
    p2[:, 1] = p[:, 1] / scale
    p2[:, 2] = p[:, 2] / scale
    return p2


def taperY(p, y1, y2, s1, s2):
    p2 = p.clone()
    #scale = torch.zeros((p.shape[0], 1))
    scale = torch.zeros(p.shape[0])
    scale[:] = s1
    idx = (p[:, 1] > y2)
    scale[idx] = s2
    idx = (p[:, 1] >= y1) & (p[:, 1] <= y2)
    t = (p[:, 1] - y1) / (y2 - y1)
    scale[idx] = (1.0 - t[idx]) * s1 + t[idx] * s2
    p2[:, 0] = p[:, 0] / scale
    p2[:, 2] = p[:, 2] / scale
    return p2


def taperZ(p, z1, z2, s1, s2):
    p2 = p.clone()
    #scale = torch.zeros((p.shape[0], 1))
    scale = torch.zeros(p.shape[0])
    scale[:] = s1
    idx = (p[:, 2] > z2)
    scale[idx] = s2
    idx = (p[:, 2] >= z1) & (p[:, 2] <= z2)
    t = (p[:, 2] - z1) / (z2 - z1)
    scale[idx] = (1.0 - t[idx]) * s1 + t[idx] * s2
    p2[:, 0] = p[:, 0] / scale
    p2[:, 1] = p[:, 1] / scale
    return p2


def rep(p, c):
    if not torch.is_tensor(c):
        c = torch.tensor(c)
    q = torch.fmod(p + 0.5 * c, c) - 0.5 * c
    # or
    #q = torch.remainder(p + 0.5*c, c)-0.5*c
    return q


def boundBlendUnion(f1, f2, f3, a0, a1, a2, a3):
    r1 = f1**2 / a1**2 + f2**2 / a2**2
    tmp_f3 = f3**2 / a3**2
    r2 = torch.zeros_like(f1)
    f3p_idx = f3 > 0.0
    r2[f3p_idx] = tmp_f3[f3p_idx]
    rr = torch.zeros_like(f1)
    r1p_idx = r1 > 0.0
    rr[r1p_idx] = r1[r1p_idx] / (r1[r1p_idx] + r2[r1p_idx])
    d = torch.zeros_like(f1)
    rr_idx = rr < 1.0
    d[rr_idx] = a0 * (1.0 - rr[rr_idx])**3 / (1.0 + rr[rr_idx])
    return f1 + f2 + torch.sqrt(f1**2 + f2**2) + d


# T is the period of the wave
# Output is in [-1,1]
def sawtooth(p, T):
    if not torch.is_tensor(T):
        T = torch.tensor(T)
    return 2.0 * (0.5 -
                  torch.atan(1.0 / torch.tan(math.pi *
                                             (p / T - 0.5))) / math.pi) - 1.0


# T is the period of the triangle wave
# Output is in [-1,1]
def triangleWave(p, T):
    if not torch.is_tensor(T):
        T = torch.tensor(T)
    return 2.0 / math.pi * torch.asin(torch.sin(math.pi * (2.0 * p / T - 0.5)))
