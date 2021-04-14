'''
Replication of cells using the sawtooth function.
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *


import polyscope as ps


def model(p):
    x0 = p[:,0]
    x1 = p[:,1]
    x2 = p[:,2]
    sp = 1.0 - (x0/10.0)**2 - (x1/10.0)**2 - (x2/10.0)**2
    norsp = sp
    offset = 0.15
    offsp = norsp - offset
    sp_shell = difference(sp, offsp)
    t1 = norsp
    scale = 9.0 * (1.0-t1) + 3.0 * t1
    xt = x0 * scale
    yt = x1 * scale
    zt = x2 * scale
    tsin = 0.75 * (1.0 - t1) + 0.85 * t1
    xslabs = torch.sin(xt) - tsin
    yslabs = torch.sin(yt) - tsin
    zslabs = torch.sin(zt) - tsin
    xrods = intersection(yslabs, zslabs)
    yrods = intersection(xslabs, zslabs)
    zrods = intersection(xslabs, yslabs)
    grid = union(blendUnion(xrods, zrods, 1.0, 3.0, 3.0), yrods)
    sphere_grid = intersection(union(sp_shell, intersection(sp, grid)), x1)
    return sphere_grid


print('Generating simple model')
v,f,n = evalToMesh(model, grid_min=(-11.0,-11.0,-11.0), grid_max=(11.0,11.0,11.0), grid_res=(64,64,64))
print('Done')

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
