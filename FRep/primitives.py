#import numpy as np
import torch
import math


def sphere(p, center, r):
    x0 = p[:, 0] - center[0]
    x1 = p[:, 1] - center[1]
    x2 = p[:, 2] - center[2]
    return r**2 - x0**2 - x1**2 - x2**2


def ellipsoid(p, center, a, b, c):
    x0 = (p[:, 0] - center[0]) / a
    x1 = (p[:, 1] - center[1]) / b
    x2 = (p[:, 2] - center[2]) / c
    return 1.0 - x0**2 - x1**2 - x2**2


def cylX(p, center, r):
    x1 = p[:, 1] - center[1]
    x2 = p[:, 2] - center[2]
    return r**2 - x1**2 - x2**2


def cylY(p, center, r):
    x0 = p[:, 0] - center[0]
    x2 = p[:, 2] - center[2]
    return r**2 - x0**2 - x2**2


def cylZ(p, center, r):
    x0 = p[:, 0] - center[0]
    x1 = p[:, 1] - center[1]
    return r**2 - x0**2 - x1**2


def ellCylX(p, center, a, b):
    x1 = (p[:, 1] - center[1]) / a
    x2 = (p[:, 2] - center[2]) / b
    return 1.0 - x1**2 - x2**2


def ellCylY(p, center, a, b):
    x0 = (p[:, 0] - center[0]) / a
    x2 = (p[:, 2] - center[2]) / b
    return 1.0 - x0**2 - x2**2


def ellCylZ(p, center, a, b):
    x0 = (p[:, 0] - center[0]) / a
    x1 = (p[:, 1] - center[1]) / b
    return 1.0 - x0**2 - x1**2


def torusX(p, center, R, r):
    x0 = p[:, 0] - center[0]
    x1 = p[:, 1] - center[1]
    x2 = p[:, 2] - center[2]
    return r**2 - x0**2 - x1**2 - x2**2 - R**2 + 2.0 * R * torch.sqrt(x1**2 +
                                                                      x2**2)


def torusY(p, center, R, r):
    x0 = p[:, 0] - center[0]
    x1 = p[:, 1] - center[1]
    x2 = p[:, 2] - center[2]
    return r**2 - x0**2 - x1**2 - x2**2 - R**2 + 2.0 * R * torch.sqrt(x0**2 +
                                                                      x2**2)


def torusZ(p, center, R, r):
    x0 = p[:, 0] - center[0]
    x1 = p[:, 1] - center[1]
    x2 = p[:, 2] - center[2]
    return r**2 - x0**2 - x1**2 - x2**2 - R**2 + 2.0 * R * torch.sqrt(x0**2 +
                                                                      x1**2)


def block(p, vertex, dx, dy, dz):
    x0 = -(p[:, 0] - vertex[0]) * (p[:, 0] - (vertex[0] + dx))
    x1 = -(p[:, 1] - vertex[1]) * (p[:, 1] - (vertex[1] + dy))
    x2 = -(p[:, 2] - vertex[2]) * (p[:, 2] - (vertex[2] + dz))
    t0 = x0 + x1 - torch.sqrt(x0**2 + x1**2)
    return t0 + x2 - torch.sqrt(t0**2 + x2**2)


def blobby(p, x0, y0, z0, a, b, T):
    s = 0.0
    for i in range(len(x0)):
        xx0 = p[:, 0] - x0[i]
        xx1 = p[:, 1] - y0[i]
        xx2 = p[:, 2] - z0[i]
        r = xx0**2 + xx1**2 + xx2**2
        s = s + b[i] * torch.exp(-a[i] * r)
    return s - T


def metaBall(p, p0, y0, z0, b, d, T):
    s = torch.zeros((p.shape[0], 1))
    for i in range(len(x0)):
        xx0 = p[:, 0] - x0[i]
        xx1 = p[:, 1] - y0[i]
        xx2 = p[:, 2] - z0[i]
        r = xx0**2 + xx1**2 + xx2**2

        idx1 = (r <= d[i] / 3.0)
        idx2 = (r <= d[i]) & ~idx1
        s[idx1] = s[idx1] + b[i] * (1.0 - 3.0 * r / d[i] * d[i])
        s[idx2] = s[idx2] + 1.5 * b[i] * (1.0 - r / d[i])**2

    return s - T


def soft(p, x0, y0, z0, d, T):
    s = torch.zeros((p.shape[0], 1))
    for i in range(len(x0)):
        xx0 = p[:, 0] - x0[i]
        xx1 = p[:, 1] - y0[i]
        xx2 = p[:, 2] - z0[i]
        r = xx0**2 + xx1**2 + xx2**2
        d2 = (d[i])**2

        s = s + 1.0 - (22.0 * r) / (9.0 * d2) + (17.0 * r**2) / (
            9.0 * d2**2) - (4.0 * r**3) / (9.0 * d2**3)

    return s - T


def coneX(p, center, R):
    xx = p[:, 0] - center[0]
    yy = (p[:, 1] - center[1]) / R
    zz = (p[:, 2] - center[2]) / R
    return xx**2 - yy**2 - zz**2


def coneY(p, center, R):
    xx = (p[:, 0] - center[0]) / R
    yy = p[:, 1] - center[1]
    zz = (p[:, 2] - center[2]) / R
    return yy**2 - xx**2 - zz**2


def coneZ(p, center, R):
    xx = (p[:, 0] - center[0]) / R
    yy = (p[:, 1] - center[1]) / R
    zz = p[:, 2] - center[2]
    return zz**2 - xx**2 - yy**2


def ellConeX(p, center, a, b):
    xx = p[:, 0] - center[0]
    yy = (p[:, 1] - center[1]) / a
    zz = (p[:, 2] - center[2]) / b
    return xx**2 - yy**2 - zz**2


def ellConeY(p, center, a, b):
    xx = (p[:, 0] - center[0]) / a
    yy = p[:, 1] - center[1]
    zz = (p[:, 2] - center[2]) / b
    return yy**2 - xx**2 - zz**2


def ellConeZ(p, center, a, b):
    xx = (p[:, 0] - center[0]) / a
    yy = (p[:, 1] - center[1]) / b
    zz = p[:, 2] - center[2]
    return zz**2 - xx**2 - yy**2


def mapBlob(p, fobj, x0, y0, z0, fobj0, sigma):
    s = torch.zeros((p.shape[0], 1))
    s[:] = fobj
    for i in range(len(x0)):
        xt = p[:, 0] - x0[i]
        yt = p[:, 1] - y0[i]
        zt = p[:, 2] - z0[i]
        d2 = (xt**2 + yt**2 + zt**2) / (sigma[i]**2 + 1e-9)
        s = s - fobj0[i] / (1.0 + d2)

    return s


def noiseG(p, amp, freq, phase):
    a1 = amp
    a2 = freq
    xt = p[:, 0]
    yt = p[:, 1]
    zt = p[:, 2]
    a2x = a2 * xt
    a2y = a2 * yt
    a2z = a2 * zt
    sx = torch.sin(a2x)
    sy = torch.sin(a2y)
    sz = torch.sin(a2z)
    a1d = a1 / 1.17
    sx2 = a1d * torch.sin(a2x / 1.35 + phase * sz)
    sy2 = a1d * torch.sin(a2y / 1.35 + phase * sx)
    sz2 = a1d * torch.sin(a2z / 1.35 + phase * sy)
    serx = a1 * sx + sx2
    sery = a1 * sy + sy2
    serz = a1 * sz + sz2
    ss = serx * sery * serz
    return ss


def superEll(p, center, a, b, c, s1, s2):
    xt = (p[:, 0] - center[0]) / a
    yt = (p[:, 1] - center[1]) / b
    zt = (p[:, 2] - center[2]) / c

    pp = 2.0 / s2
    xp = (torch.abs(xt))**p
    yp = (torch.abs(yt))**p
    zp = (torch.abs(zt))**(2.0 / s1)

    xyp = (xp + yp)**(s2 / s1)
    return 1.0 - xyp - zp


def gyroid(p, alpha, beta, gamma, c):
    X = 2.0 * alpha * math.pi * p[:, 0]
    Y = 2.0 * beta * math.pi * p[:, 1]
    Z = 2.0 * gamma * math.pi * p[:, 2]
    return c - torch.sin(X) * torch.cos(Y) - torch.sin(Y) * torch.cos(
        Z) - torch.sin(Z) * torch.cos(X)


def gyroid_sheet(p, alpha, beta, gamma, c1, c2):
    g1 = gyroid(p, alpha, beta, gamma, c1)
    g2 = gyroid(p, alpha, beta, gamma, c2)
    # R-function based difference
    return g1 - g2 - torch.sqrt(g1**2 + g2**2)


def convPoint(p, vect, S, T):
    '''
    p - point coordinates
    vect - array of skeleton points' coordinates
    S - array of kernel width control parameters for each point
    T - threshold
    '''

    if not torch.is_tensor(vect):
        vect = torch.tensor(vect)
    if not torch.is_tensor(S):
        S = torch.tensor(S)

    X = p[:, 0]
    Y = p[:, 1]
    Z = p[:, 2]

    f = 0.0
    # number of points
    N = len(S)
    for n in range(N):
        pointX = vect[3 * n + 0]
        pointY = vect[3 * n + 1]
        pointZ = vect[3 * n + 2]
        r2 = (pointX - X)**2 + (pointY - Y)**2 + (pointZ - Z)**2
        kernelS = S[n]
        f = f + 1.0 / (1.0 + kernelS**2 * r2)**2

    return f - T


def convLine(p, begin, end, S, T):
    '''  
    Primitive: Аnalytical convolution for a segment with Cauchy kernel 
    [McCormack and Sherstyuk 1998]
    Definition:  1 / (1 + S^2*R^2)^2
                 R is the distance between primitive and x
    Parameters:  
                 p - points coordinate array
                 begin - beginning points coordinate array
                 end - ending points coordinate array
                 S - control value for width of the kernel
                 T - threshold value
    '''

    if not torch.is_tensor(begin):
        begin = torch.tensor(begin)

    if not torch.is_tensor(end):
        end = torch.tensor(end)

    if not torch.is_tensor(S):
        S = torch.tensor(S)

    X = p[:, 0]
    Y = p[:, 1]
    Z = p[:, 2]

    f = 0.0

    # the number of primitive
    N = len(S)
    for n in range(0, N):
        l = torch.sqrt((end[3 * n] - begin[3 * n])**2 +
                       (end[3 * n + 1] - begin[3 * n + 1])**2 +
                       (end[3 * n + 2] - begin[3 * n + 2])**2)

        if l == 0.0:
            return 0

        # normalized vector from beginnig to ending  Point
        ax = (end[3 * n] - begin[3 * n]) / l
        ay = (end[3 * n + 1] - begin[3 * n + 1]) / l
        az = (end[3 * n + 2] - begin[3 * n + 2]) / l
        # d = r - b
        dx = X - begin[3 * n]
        dy = Y - begin[3 * n + 1]
        dz = Z - begin[3 * n + 2]

        xx = dx * ax + dy * ay + dz * az
        p = torch.sqrt(1 + S[n] * S[n] *
                       (dx * dx + dy * dy + dz * dz - xx * xx))
        q = torch.sqrt(1 + S[n] * S[n] *
                       (dx * dx + dy * dy + dz * dz + l * l - 2 * l * xx))

        f += xx / (2.0 * p * p * (p * p + S[n] * S[n] * xx * xx)) + (
            l - xx) / (2.0 * p * p * q * q) + (
                torch.atan(S[n] * xx / p) +
                torch.atan(S[n] * (l - xx) / p)) / (2.0 * S[n] * p * p * p)

    return f - T


def convCurve(p, vect, S, T):
    '''
    Primitive: Cauchy Curve (connected line) with Convolution Surface
    Definition:  1 / (1 + S^2*R^2)^2
                 R is the distance between primitive and x
    Parameters:  
                 p - points coordinate array
                 vect[n] - beginning points coordinate array
                 vect[n+1] - ending points coordinate array
                 S - control value for width of the kernel
                 T - threshold value
    '''

    if not torch.is_tensor(vect):
        vect = torch.tensor(vect)

    X = p[:, 0]
    Y = p[:, 1]
    Z = p[:, 2]

    f = 0.0

    # the number of primitive
    N = len(S)
    for n in range(0, N):
        l = torch.sqrt((vect[3 * (n + 1)] - vect[3 * n])**2 +
                       (vect[3 * (n + 1) + 1] - vect[3 * n + 1])**2 +
                       (vect[3 * (n + 1) + 2] - vect[3 * n + 2])**2)

        if l == 0.0:
            return 0

        # normalized vector from beginnig to ending point
        ax = (vect[3 * (n + 1)] - vect[3 * n]) / l
        ay = (vect[3 * (n + 1) + 1] - vect[3 * n + 1]) / l
        az = (vect[3 * (n + 1) + 2] - vect[3 * n + 2]) / l
        # d = r - b
        dx = X - vect[3 * n]
        dy = Y - vect[3 * n + 1]
        dz = Z - vect[3 * n + 2]

        xx = dx * ax + dy * ay + dz * az
        p = torch.sqrt(1 + S[n] * S[n] *
                       (dx * dx + dy * dy + dz * dz - xx * xx))
        q = torch.sqrt(1 + S[n] * S[n] *
                       (dx * dx + dy * dy + dz * dz + l * l - 2 * l * xx))

        f += xx / (2.0 * p * p * (p * p + S[n] * S[n] * xx * xx)) + (
            l - xx) / (2.0 * p * p * q * q) + (
                torch.atan(S[n] * xx / p) +
                torch.atan(S[n] * (l - xx) / p)) / (2.0 * S[n] * p * p * p)

    return f - T
