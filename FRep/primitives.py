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


def convLineR(p, begin, end, S, R):
    '''
    Primitive: Cauchy Line with Convolution Surface
    Definition:  1 / (1 + S^2*R^2)^2
                 R is the distance between primitive and x
    Parameters:
                 p - points coordinate array
                 begin - beginning points coordinate array
                 end - ending points coordinate array
                 S - control value for width of the kernel
                 R - width of primitive
    '''

    # the number of primitive
    N = len(end)
    N = int(N / 3)  # or N//3

    if not torch.is_tensor(begin):
        begin = torch.tensor(begin)

    if not torch.is_tensor(end):
        end = torch.tensor(end)

    X = p[:, 0]
    Y = p[:, 1]
    Z = p[:, 2]

    #f = 0.0
    f = torch.zeros_like(X)

    if R < 1.0:
        S = 1.0 / R

    for n in range(0, N):
        l = torch.sqrt((end[3 * n] - begin[3 * n])**2 +
                       (end[3 * n + 1] - begin[3 * n + 1])**2 +
                       (end[3 * n + 2] - begin[3 * n + 2])**2)

        if l == 0.0:
            return 0.0

        # normalized vector from beginnig to ending  Point
        ax = (end[3 * n] - begin[3 * n]) / l
        ay = (end[3 * n + 1] - begin[3 * n + 1]) / l
        az = (end[3 * n + 2] - begin[3 * n + 2]) / l
        # d = r - b
        dx = X - begin[3 * n]
        dy = Y - begin[3 * n + 1]
        dz = Z - begin[3 * n + 2]

        #pT = torch.sqrt(1.0 + (S**2) * (R**2))
        pT = math.sqrt(1.0 + (S**2) * (R**2))
        qT2 = 1.0 + (S**2) * ((R**2) + (l**2))

        T = l / (2.0 *
                 (pT**2) * qT2) + (torch.atan(S * l / pT)) / (2.0 * S *
                                                              (pT**2) * pT)
        xx = dx * ax + dy * ay + dz * az
        p = torch.sqrt(1.0 + (S**2) * ((dx**2) + (dy**2) + (dz**2) - (xx**2)))
        q2 = 1.0 + (S**2) * ((dx**2) + (dy**2) + (dz**2) +
                             (l**2) - 2.0 * l * xx)

        f_tmp = xx / (2.0 * p * p * (p * p + S * S * xx * xx)) + (l - xx) / (2.0 * p * p * q2) + \
                (torch.atan(S * xx / p) + torch.atan(S * (l - xx) / p)) / (2.0 * S * p * p * p)

        #  if f < 1.0:

        lt1 = f < 1.0
        f[lt1] = f[lt1] + f_tmp[lt1]
        gt1 = f > 1.0
        f[gt1] = 1.0

        # if torch.lt(f,1.0).all():
        #     f = f + f_tmp
        #     if torch.gt(f,1.0).all():
        #         f = 1.0

    return f - T


def convTriangle(p, vect, S, T):
    '''
    Primitive: Cauchy Triangle with Convolution Surface
    Definition:  1 / (1 + S^2*R^2)^2
    R is the distance between primitive and x
    Parameters:
                p - points coordinate array
                vect - triangle coordinate array
                S - control value for width of the kernel
                T - threshold value
    '''

    length = [0, 0, 0]
    # length[0] means distance of coordinates 1 and 2
    # length[1] means distance of coordinates 2 and 3
    # length[2] means distance of coordinates 3 and 1

    if not torch.is_tensor(vect):
        vect = torch.tensor(vect)

    X = p[:, 0]
    Y = p[:, 1]
    Z = p[:, 2]

    f = 0.0

    # the number of primitive
    N = len(S)
    for n in range(N):
        a1x = vect[9 * n]
        a1y = vect[9 * n + 1]
        a1z = vect[9 * n + 2]
        a2x = vect[9 * n + 3]
        a2y = vect[9 * n + 4]
        a2z = vect[9 * n + 5]
        a3x = vect[9 * n + 6]
        a3y = vect[9 * n + 7]
        a3z = vect[9 * n + 8]

        length[0] = torch.sqrt((a2x - a1x)**2 + (a2y - a1y)**2 +
                               (a2z - a1z)**2)
        length[1] = torch.sqrt((a3x - a2x)**2 + (a3y - a2y)**2 +
                               (a3z - a2z)**2)
        length[2] = torch.sqrt((a1x - a3x)**2 + (a1y - a3y)**2 +
                               (a1z - a3z)**2)

        if (length[1] >= length[2]) and (length[1] > length[0]):
            tempx = a1x
            tempy = a1y
            tempz = a1z
            a1x = a2x
            a1y = a2y
            a1z = a2z
            a2x = a3x
            a2y = a3y
            a2z = a3z
            a3x = tempx
            a3y = tempy
            a3z = tempz
        elif (length[2] >= length[1]) and (length[2] > length[0]):
            tempx = a1x
            tempy = a1y
            tempz = a1z
            a1x = a3x
            a1y = a3y
            a1z = a3z
            a3x = a2x
            a3y = a2y
            a3z = a2z
            a2x = tempx
            a2y = tempy
            a2z = tempz

        length[0] = torch.sqrt((a2x - a1x)**2 + (a2y - a1y)**2 +
                               (a2z - a1z)**2)
        length[1] = torch.sqrt((a3x - a2x)**2 + (a3y - a2y)**2 +
                               (a3z - a2z)**2)
        length[2] = torch.sqrt((a1x - a3x)**2 + (a1y - a3y)**2 +
                               (a1z - a3z)**2)

        a21x = a2x - a1x
        a21y = a2y - a1y
        a21z = a2z - a1z
        a13x = a1x - a3x
        a13y = a1y - a3y
        a13z = a1z - a3z

        t = -(a21x * a13x + a21y * a13y + a21z * a13z) / (length[0])**2
        bx = a1x + t * a21x
        by = a1y + t * a21y
        bz = a1z + t * a21z

        dx = X - bx
        dy = Y - by
        dz = Z - bz

        ux = a2x - bx
        uy = a2y - by
        uz = a2z - bz
        ul = torch.sqrt((ux)**2 + (uy)**2 + (uz)**2)
        ux = ux / ul
        uy = uy / ul
        uz = uz / ul

        vx = a3x - bx
        vy = a3y - by
        vz = a3z - bz
        vl = torch.sqrt((vx)**2 + (vy)**2 + (vz)**2)
        vx = vx / vl
        vy = vy / vl
        vz = vz / vl

        d2 = (dx)**2 + (dy)**2 + (dz)**2
        u = dx * ux + dy * uy + dz * uz
        v = dx * vx + dy * vy + dz * vz
        h = torch.sqrt((a3x - bx)**2 + (a3y - by)**2 + (a3z - bz)**2)
        a1 = torch.sqrt((a1x - bx)**2 + (a1y - by)**2 + (a1z - bz)**2)
        a2 = torch.sqrt((a2x - bx)**2 + (a2y - by)**2 + (a2z - bz)**2)

        g = v - h
        m = a2 * g + u * h
        k = u * h - a1 * g
        C2 = 1.0 / ((S[n]**2)) + d2 - (u)**2
        C = torch.sqrt(C2)
        q = C2 - (v)**2
        w = C2 - 2.0 * v * h + (h)**2
        A2 = ((a1)**2) * w + ((h)**2) * (q + (u)**2) - 2.0 * a1 * h * u * g
        A = torch.sqrt(A2)
        B2 = ((a2)**2) * w + ((h)**2) * (q + (u)**2) + 2.0 * a2 * h * u * g
        B = torch.sqrt(B2)

        n1 = a1 + u
        n2 = a2 - u
        n3 = a1 * n1 + v * h
        n4 = -a1 * u - g * h
        n5 = -a2 * n2 - v * h
        n6 = -a2 * u + g * h

        arc1 = k * (torch.atan(n3 / A) + torch.atan(n4 / A)) / A
        arc2 = m * (torch.atan(n5 / B) + torch.atan(n6 / B)) / B
        arc3 = v * (torch.atan(n1 / C) + torch.atan(n2 / C)) / C
        f += (arc1 + arc2 + arc3) / (2.0 * q * S[n])

    return f - T


def convMesh(p, vect, tri, S, T):
    '''
    Primitive: Cauchy Mesh (connected triangle) with Convolution Surface
    Definition:  1 / (1 + S^2*R^2)^2
                 R is the distance between primitive and x
    Parameters:
                 p - points coordinate array
                 vect - triangle coordinate array
                 tri - index of each triangle coordinate
                 S - control value for width of the kernel
                 T - threshold value
    '''

    length = [0, 0, 0]
    # length[0] means distance of coordinates 1 and 2
    # length[1] means distance of coordinates 2 and 3
    # length[2] means distance of coordinates 3 and 1

    if not torch.is_tensor(vect):
        vect = torch.tensor(vect)

    X = p[:, 0]
    Y = p[:, 1]
    Z = p[:, 2]

    f = 0.0

    # the number of primitive
    N = len(S)
    for n in range(N):
        #triangle coodinates
        a1x = vect[3 * (tri[3 * n] - 1)]
        a1y = vect[3 * (tri[3 * n] - 1) + 1]
        a1z = vect[3 * (tri[3 * n] - 1) + 2]
        a2x = vect[3 * (tri[3 * n + 1] - 1)]
        a2y = vect[3 * (tri[3 * n + 1] - 1) + 1]
        a2z = vect[3 * (tri[3 * n + 1] - 1) + 2]
        a3x = vect[3 * (tri[3 * n + 2] - 1)]
        a3y = vect[3 * (tri[3 * n + 2] - 1) + 1]
        a3z = vect[3 * (tri[3 * n + 2] - 1) + 2]

        length[0] = torch.sqrt((a2x - a1x)**2 + (a2y - a1y)**2 +
                               (a2z - a1z)**2)
        length[1] = torch.sqrt((a3x - a2x)**2 + (a3y - a2y)**2 +
                               (a3z - a2z)**2)
        length[2] = torch.sqrt((a1x - a3x)**2 + (a1y - a3y)**2 +
                               (a1z - a3z)**2)

        if (length[1] >= length[2]) and (length[1] > length[0]):
            tempx = a1x
            tempy = a1y
            tempz = a1z
            a1x = a2x
            a1y = a2y
            a1z = a2z
            a2x = a3x
            a2y = a3y
            a2z = a3z
            a3x = tempx
            a3y = tempy
            a3z = tempz
        elif (length[2] >= length[1]) and (length[2] > length[0]):
            tempx = a1x
            tempy = a1y
            tempz = a1z
            a1x = a3x
            a1y = a3y
            a1z = a3z
            a3x = a2x
            a3y = a2y
            a3z = a2z
            a2x = tempx
            a2y = tempy
            a2z = tempz

        length[0] = torch.sqrt((a2x - a1x)**2 + (a2y - a1y)**2 +
                               (a2z - a1z)**2)
        length[1] = torch.sqrt((a3x - a2x)**2 + (a3y - a2y)**2 +
                               (a3z - a2z)**2)
        length[2] = torch.sqrt((a1x - a3x)**2 + (a1y - a3y)**2 +
                               (a1z - a3z)**2)

        a21x = a2x - a1x
        a21y = a2y - a1y
        a21z = a2z - a1z
        a13x = a1x - a3x
        a13y = a1y - a3y
        a13z = a1z - a3z

        t = -(a21x * a13x + a21y * a13y + a21z * a13z) / (length[0])**2
        bx = a1x + t * a21x
        by = a1y + t * a21y
        bz = a1z + t * a21z

        dx = X - bx
        dy = Y - by
        dz = Z - bz

        ux = a2x - bx
        uy = a2y - by
        uz = a2z - bz
        ul = torch.sqrt((ux)**2 + (uy)**2 + (uz)**2)
        ux = ux / ul
        uy = uy / ul
        uz = uz / ul

        vx = a3x - bx
        vy = a3y - by
        vz = a3z - bz
        vl = torch.sqrt((vx)**2 + (vy)**2 + (vz)**2)
        vx = vx / vl
        vy = vy / vl
        vz = vz / vl

        d2 = (dx)**2 + (dy)**2 + (dz)**2
        u = dx * ux + dy * uy + dz * uz
        v = dx * vx + dy * vy + dz * vz
        h = torch.sqrt((a3x - bx)**2 + (a3y - by)**2 + (a3z - bz)**2)
        a1 = torch.sqrt((a1x - bx)**2 + (a1y - by)**2 + (a1z - bz)**2)
        a2 = torch.sqrt((a2x - bx)**2 + (a2y - by)**2 + (a2z - bz)**2)

        g = v - h
        m = a2 * g + u * h
        k = u * h - a1 * g
        C2 = 1.0 / ((S[n]**2)) + d2 - (u)**2
        C = torch.sqrt(C2)
        q = C2 - (v)**2
        w = C2 - 2.0 * v * h + (h)**2
        A2 = ((a1)**2) * w + ((h)**2) * (q + (u)**2) - 2.0 * a1 * h * u * g
        A = torch.sqrt(A2)
        B2 = ((a2)**2) * w + ((h)**2) * (q + (u)**2) + 2.0 * a2 * h * u * g
        B = torch.sqrt(B2)

        n1 = a1 + u
        n2 = a2 - u
        n3 = a1 * n1 + v * h
        n4 = -a1 * u - g * h
        n5 = -a2 * n2 - v * h
        n6 = -a2 * u + g * h

        arc1 = k * (torch.atan(n3 / A) + torch.atan(n4 / A)) / A
        arc2 = m * (torch.atan(n5 / B) + torch.atan(n6 / B)) / B
        arc3 = v * (torch.atan(n1 / C) + torch.atan(n2 / C)) / C
        f += (arc1 + arc2 + arc3) / (2.0 * q * S[n])

    return f - T


def convArc(p, center, radius, theta, axis, angle, S, T):
    '''
    Primitive: Convolution with set of skeleton arcs
    Definition:  1 / (1 + S^2*R^2)^2
                 R is the distance between primitive and x
    Parameters:
                 p - points coordinate array
                 angle - angles of rotation for arcs around axis of rotation
                 axis - array of vectors defining axis of rotation for each arc (placed on local xy-plane)
                 theta -  array of arcs' angles (from positive x-axis counter-clockwise, 360 for full circle)
                 radius - arc radius
                 center - center of arc
                 S - control value for width of the kernel
                 T - threshold value
    '''

    if not torch.is_tensor(center):
        center = torch.tensor(center)

    if not torch.is_tensor(radius):
        radius = torch.tensor(radius)

    if not torch.is_tensor(theta):
        theta = torch.tensor(theta)

    if not torch.is_tensor(axis):
        axis = torch.tensor(axis)

    if not torch.is_tensor(angle):
        angle = torch.tensor(angle)

    if not torch.is_tensor(S):
        S = torch.tensor(S)

    #PI = 3.141592
    PI = math.pi
    rd = PI / 180.0
    over_i = 0.0
    over_j = 0.0
    over_k = 1.0

    EPS = 0.01

    X = p[:, 0]
    Y = p[:, 1]
    Z = p[:, 2]

    f = torch.zeros_like(X)

    # the number of primitive
    N = len(S)
    for n in range(N):
        cx = center[3 * n]  # center of arc
        cy = center[3 * n + 1]
        cz = center[3 * n + 2]

        r = radius[n]
        angle[n] += EPS  # avoid error

        i = axis[3 * n] + EPS  # avoid error
        j = axis[3 * n + 1] + EPS  # avoid error
        k = axis[3 * n + 2] + EPS  # avoid error

        # if not torch.is_tensor(i):
        #     i = torch.tensor(i)
        # if not torch.is_tensor(j):
        #     j = torch.tensor(j)
        # if not torch.is_tensor(k):
        #     k = torch.tensor(k)

        length = torch.sqrt(i * i + j * j + k * k)
        if length < EPS:
            length = EPS

        i /= length  # calculate normal vector around which arc rotates
        j /= length
        k /= length

        c = torch.cos(rd * (-1.0 * angle[n]))
        s = torch.sin(rd * (-1.0 * angle[n]))

        one_c = 1.0 - c

        ii = i * i
        jj = j * j
        kk = k * k
        ij = i * j
        jk = j * k
        ki = k * i
        is_ = i * s
        js = j * s
        ks = k * s

        if theta[n] > 360.0:
            theta[n] = 360.0

        # [Begin] over PI operation
        if theta[n] > 180.0:
            over_th = (theta[n] - 180.0) * rd
            theta[n] = 180.0

            # rotate by -angle
            tempx = (c + ii * one_c) * (X - cx) + (-ks + ij * one_c) * (
                Y - cy) + (js + ki * one_c) * (Z - cz)
            tempy = (ks + ij * one_c) * (X - cx) + (c + jj * one_c) * (
                Y - cy) + (-is_ + jk * one_c) * (Z - cz)
            tempz = (-js + ki * one_c) * (X - cx) + (is_ + jk * one_c) * (
                Y - cy) + (c + kk * one_c) * (Z - cz)

            # [Begin] rotate -PI operation
            #over_c = torch.cos(rd * (-180.0))
            over_c = math.cos(rd * (-180.0))
            #over_s = torch.sin(rd * (-180.0))
            over_s = math.sin(rd * (-180.0))
            over_one_c = 1.0 - over_c

            over_ii = (over_i)**2
            over_jj = (over_j)**2
            over_kk = (over_k)**2
            over_ij = over_i * over_j
            over_jk = over_j * over_k
            over_ki = over_k * over_i
            over_is = over_i * over_s
            over_js = over_j * over_s
            over_ks = over_k * over_s

            over_x = (over_c + over_ii * over_one_c) * (tempx) + (
                -over_ks + over_ij * over_one_c) * (tempy) + (
                    over_js + over_ki * over_one_c) * (tempz)
            over_y = (over_ks + over_ij * over_one_c) * (tempx) + (
                over_c + over_jj * over_one_c) * (tempy) + (
                    -1 * over_is + over_jk * over_one_c) * (tempz)
            over_z = (-1 * over_js + over_ki * over_one_c) * (tempx) + (
                over_is + over_jk * over_one_c) * (tempy) + (
                    over_c + over_kk * over_one_c) * (tempz)
            # [End] rotate -PI operation

            a = 2.0 * r * S[n] * S[n]
            d2 = (over_x)**2 + (over_y)**2 + (over_z)**2
            b = 1.0 + (r)**2 * (S[n])**2 + ((S[n])**2) * d2
            p2 = -1 * (r)**4 * (S[n])**4 + 2.0 * (r)**2 * (S[n])**2 * (
                (S[n])**2 * (d2 - 2.0 *
                             (over_z)**2) - 1.0) - (1.0 + (S[n])**2 * d2)**2

            # if p2 < 0.0:
            #     p1 = torch.sqrt(-1*p2)
            # else:
            #     p1 = torch.sqrt(p2);

            p1 = p2
            p2neg_idx = p2 < 0.0
            p1[p2neg_idx] = torch.sqrt(-p2[p2neg_idx])
            p1[~p2neg_idx] = torch.sqrt(p2[~p2neg_idx])

            p3 = p1 * p2

            f1 = (b * over_y) / (over_x * p2 * (a * over_x - b)) + (a * (
                (over_x)**2 +
                (over_y)**2) * torch.sin(over_th) - b * over_y) / (
                    over_x * p2 * (a * (over_x * torch.cos(over_th) +
                                        over_y * torch.sin(over_th)) - b))

            # if torch.lt(p2, 0.0).all():
            #     f2 = 2.0 * b * (torch.atan(-a * over_y / p1) + torch.atan((a * over_y - (a * over_x + b) * torch.tan(over_th / 2.0)) / p1)) / p3;
            # else:
            #     f2 = 2.0 * b * (torch.atanh(a * over_y / p1) + torch.atanh(((a * over_x + b) * torch.tan(over_th / 2.0) - a * over_y) / p1)) / p3;

            f2 = f1
            f2[p2neg_idx] = 2.0 * b[p2neg_idx] * (torch.atan(
                -a * over_y[p2neg_idx] / p1[p2neg_idx]) + torch.atan(
                    (a * over_y[p2neg_idx] -
                     (a * over_x[p2neg_idx] + b[p2neg_idx]) * torch.tan(
                         over_th / 2.0)) / p1[p2neg_idx])) / p3[p2neg_idx]
            f2[~p2neg_idx] = 2.0 * b[~p2neg_idx] * (torch.atanh(
                a * over_y[~p2neg_idx] / p1[~p2neg_idx]) + torch.atanh(
                    ((a * over_x[~p2neg_idx] + b[~p2neg_idx]) *
                     torch.tan(over_th / 2.0) - a * over_y[~p2neg_idx]) /
                    p1[~p2neg_idx])) / p3[~p2neg_idx]

            f += f1 + f2


#-------------
        th = theta[n] * rd
        new_x = (c + ii * one_c) * (X - cx) + (-1.0 * ks + ij * one_c) * (
            Y - cy) + (js + ki * one_c) * (Z - cz)
        new_y = (ks + ij * one_c) * (X - cx) + (c + jj * one_c) * (Y - cy) + (
            -1.0 * is_ + jk * one_c) * (Z - cz)
        new_z = (-1.0 * js + ki * one_c) * (X - cx) + (is_ + jk * one_c) * (
            Y - cy) + (c + kk * one_c) * (Z - cz)

        a = 2.0 * r * S[n] * S[n]
        d2 = (new_x)**2 + (new_y)**2 + (new_z)**2
        b = 1.0 + (r)**2 * (S[n])**2 + (S[n])**2 * d2
        p2 = -1.0 * (r)**4 * (S[n])**4 + 2.0 * (r)**2 * (S[n])**2 * (
            (S[n])**2 * (d2 - 2.0 * (new_z)**2) - 1.0) - (1.0 +
                                                          (S[n])**2 * d2)**2

        # if torch.lt(p2, 0.0).all():
        #     p1 =  torch.sqrt(-1*p2)
        # else:
        #     p1 = torch.sqrt(p2);

        p1 = p2
        p2neg_idx = p2 < 0.0
        p1[p2neg_idx] = torch.sqrt(-1.0 * p2[p2neg_idx])
        p1[~p2neg_idx] = torch.sqrt(p2[~p2neg_idx])

        p3 = p1 * p2

        f1 = (b * new_y) / (new_x * p2 * (a * new_x - b)) + (a * (
            (new_x)**2 + (new_y)**2) * torch.sin(th) - b * new_y) / (
                new_x * p2 *
                (a * (new_x * torch.cos(th) + new_y * torch.sin(th)) - b))

        # if torch.lt(p2, 0.0).all():
        #     f2 = 2.0 * b * (torch.atan(-a * new_y / p1) + torch.atan((a * new_y - (a * new_x + b) * torch.tan(th / 2.0)) / p1)) / p3;
        # else:
        #     f2 = 2.0 * b * (torch.atanh(a * new_y / p1) + torch.atanh(((a * new_x + b) * torch.tan(th / 2.0) - a * new_y) / p1)) / p3;

        f2 = f1
        f2[p2neg_idx] = 2.0 * b[p2neg_idx] * (
            torch.atan(-a * new_y[p2neg_idx] / p1[p2neg_idx]) + torch.atan(
                (a * new_y[p2neg_idx] -
                 (a * new_x[p2neg_idx] + b[p2neg_idx]) * torch.tan(th / 2.0)) /
                p1[p2neg_idx])) / p3[p2neg_idx]
        f2[~p2neg_idx] = 2.0 * b[~p2neg_idx] * (
            torch.atanh(a * new_y[~p2neg_idx] / p1[~p2neg_idx]) + torch.atanh((
                (a * new_x[~p2neg_idx] + b[~p2neg_idx]) * torch.tan(th / 2.0) -
                a * new_y[~p2neg_idx]) / p1[~p2neg_idx])) / p3[~p2neg_idx]

        f += f1 + f2

    return f - T
