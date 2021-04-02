import numpy as np


# Non SDF primitives
# The syntax mimics the HF library

def sphere(p, center, r):
    x0 = p[:,0] - center[0]
    x1 = p[:,1] - center[1]
    x2 = p[:,2] - center[2]
    return r**2 - x0**2 - x1**2 - x2**2

def ellipsoid(p, center, a, b, c):
    x0 = (p[:,0] - center[0])/a
    x1 = (p[:,1] - center[1])/b
    x2 = (p[:,2] - center[2])/c
    return 1.0 - x0**2 - x1**2 - x2**2

def cylX(p, center, r):
    x1 = p[:,1] - center[1]
    x2 = p[:,2] - center[2]
    return r**2 - x1**2 - x2**2

def cylY(p, center, r):
    x0 = p[:,0] - center[0]
    x2 = p[:,2] - center[2]
    return r**2 - x0**2 - x2**2

def cylZ(p, center, r):
    x0 = p[:,0] - center[0]
    x1 = p[:,1] - center[1]
    return r**2 - x0**2 - x1**2

def ellCylX(p, center, a, b):
    x1 = (p[:,1] - center[1])/a
    x2 = (p[:,2] - center[2])/b
    return 1.0 - x1**2 - x2**2

def ellCylY(p, center, a, b):
    x0 = (p[:,0] - center[0])/a
    x2 = (p[:,2] - center[2])/b
    return 1.0 - x0**2 - x2**2

def ellCylZ(p, center, a, b):
    x0 = (p[:,0] - center[0])/a
    x1 = (p[:,1] - center[1])/b
    return 1.0 - x0**2 - x1**2

def torusX(p, center, R, r):
    x0 = p[:,0]-center[0]
    x1 = p[:,1]-center[1]
    x2 = p[:,2]-center[2]
    return r**2 - x0**2 - x1**2 - x2**2 - R**2 + 2.0*R*np.sqrt(x1**2+x2**2)

def torusY(p, center, R, r):
    x0 = p[:,0]-center[0]
    x1 = p[:,1]-center[1]
    x2 = p[:,2]-center[2]
    return r**2 - x0**2 - x1**2 - x2**2 - R**2 + 2.0*R*np.sqrt(x0**2+x2**2)

def torusZ(p, center, R, r):
    x0 = p[:,0]-center[0]
    x1 = p[:,1]-center[1]
    x2 = p[:,2]-center[2]
    return r**2 - x0**2 - x1**2 - x2**2 - R**2 + 2.0*R*np.sqrt(x0**2+x1**2)

def block(p, vertex, dx, dy, dz):
    x0 = -(p[:,0]-vertex[0]) * (p[:,0]-(vertex[0]+dx))
    x1 = -(p[:,1]-vertex[1]) * (p[:,1]-(vertex[1]+dy))
    x2 = -(p[:,2]-vertex[2]) * (p[:,2]-(vertex[2]+dz))
    t0 = x0 + x1 - np.sqrt(x0**2 + x1**2)
    return t0 + x2 - np.sqrt(t0**2 + x2**2)

def blobby(p, x0, y0, z0, a, b, T):
    s = 0.0
    for i in range(len(x0)):
        xx0 = p[:,0] - x0[i]
        xx1 = p[:,1] - y0[i]
        xx2 = p[:,2] - z0[i]
        r = xx0**2 + xx1**2 + xx2**2
        s = s + b[i]*np.exp(-a[i]*r)
    return s-T

