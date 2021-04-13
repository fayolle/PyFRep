import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.primitives import *
from FRep.reinit import trainPPoisson
from FRep.grid import torchLinearGrid, torchLinearSampling
from FRep.IO import saveVTK


def model(p):
    sp1 = sphere(p, center=(0,0,0), r=1)
    return sp1


modelNN = trainPPoisson(num_iters=1, fun=model, grid_min=(-2.0,-2.0,-2.0), grid_max=(2.0,2.0,2.0), p=2, device='cpu')
xyz = torchLinearGrid(grid_min=(-2.0,-2.0,-2.0), grid_max=(2.0,2.0,2.0), grid_res=(16,16,16), device='cpu')
f = torchLinearSampling(modelNN, xyz)
saveVTK('test.vtk', xyz, (16,16,16), f)

