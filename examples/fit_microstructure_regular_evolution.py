import torch

import FRep
from FRep.ops import *
from FRep.IO import readPointCloud
from FRep.fitting import train, regularizedEvolution, Evaluator
from FRep.mesh import *


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
    scale1 = param[0]
    scale2 = param[1]
    scale = scale1 * (1.0 - t1) + scale2 * t1
    xt = x0 * scale
    yt = x1 * scale
    zt = x2 * scale
    freq1 = param[2]
    freq2 = param[3]
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


try:
    # Read a point cloud
    pc = readPointCloud('data/sphere_grid2.xyz')
    xyz = pc[:, 0:3]  # x,y,z coordinates only
    xyz = torch.tensor(xyz)

    # Bounds for the parameters
    lower_bounds = [1.0, 1.0, 0.1, 0.1]
    upper_bounds = [10.0, 10.0, 2.0, 2.0]

    # Fit parameters with regularized evolution
    evaluator = Evaluator(model, xyz)
    history = regularizedEvolution(evaluator, lower_bounds, upper_bounds, cycles=10000, population_size=100, sample_size=10)

    # Best creature
    best = min(history, key=lambda i: i.score)
    print('After regularized evolution')
    print(best.params)

    # Now use SGD to further optimize the parameters
    params = train(model, lower_bounds, upper_bounds, xyz, param_init=best.params, num_iters=100)
    print('After SGD')
    print(params)

except FileNotFoundError as e:
    print(f'File not found: {e}')
except (IOError, ValueError) as e:
    print(f'Failed to read point cloud: {e}')
except Exception as e:
    print(f'Unexpected error: {e}')
