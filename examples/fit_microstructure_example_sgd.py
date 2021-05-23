import torch

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.IO import readPointCloud
from FRep.fitting import train


# Define the model
def model(p, param):
    x0 = p[:, 0]
    x1 = p[:, 1]
    x2 = p[:, 2]
    sp = 1.0 - (x0 / 10.0)**2 - (x1 / 10.0)**2 - (x2 / 10.0)**2
    norsp = sp
    offset = 0.15
    offsp = norsp - offset
    sp_shell = difference(sp, offsp)
    t1 = norsp
    scale1 = param[0]  # 9.0
    scale2 = param[1]  # 3.0
    scale = scale1 * (1.0 - t1) + scale2 * t1
    xt = x0 * scale
    yt = x1 * scale
    zt = x2 * scale
    freq1 = param[2]  # 0.75
    freq2 = param[3]  # 0.85
    tsin = freq1 * (1.0 - t1) + freq2 * t1
    xslabs = torch.sin(xt) - tsin
    yslabs = torch.sin(yt) - tsin
    zslabs = torch.sin(zt) - tsin
    xrods = intersection(yslabs, zslabs)
    yrods = intersection(xslabs, zslabs)
    zrods = intersection(xslabs, yslabs)
    grid = union(blendUnion(xrods, zrods, 1.0, 3.0, 3.0), yrods)
    sphere_grid = intersection(union(sp_shell, intersection(sp, grid)), x1)
    return sphere_grid


# Read a point cloud
# wget https://gist.githubusercontent.com/fayolle/4788f619b54ec255f1a771e1cced8369/raw/fbab656736c4af1b4aab1a483e95abb02d2de88f/sphere_grid2.xyz
pc = readPointCloud('data/sphere_grid2.xyz')
xyz = pc[:, 0:3]  # x,y,z coordinates only

# Fit with sgd
lb = [1.0, 1.0, 0.1, 0.1]
ub = [10.0, 10.0, 2.0, 2.0]
param = train(model, lb, ub, xyz, num_iters=100)
print(param)
