import FRep
from FRep.primitives import *
from FRep.reinit import trainPPoisson
from FRep.grid import torchLinearGrid, torchLinearSampling
from FRep.IO import writeVTK


def model(p):
    sp1 = sphere(p, center=(0, 0, 0), r=1)
    return sp1


modelNN = trainPPoisson(num_iters=100,
                        fun=model,
                        grid_min=(-2.0, -2.0, -2.0),
                        grid_max=(2.0, 2.0, 2.0),
                        p=2,
                        device='cpu')
xyz = torchLinearGrid(grid_min=(-2.0, -2.0, -2.0),
                      grid_max=(2.0, 2.0, 2.0),
                      grid_res=(16, 16, 16),
                      device='cpu')
f = torchLinearSampling(modelNN, xyz)

try:
    writeVTK('test.vtk', xyz, (16, 16, 16), f)
except IOError as e:
    print(f'Failed to write VTK file: {e}')
except InvalidDataError as e:
    print(f'Error: Trying to write invalid data: {e}')
