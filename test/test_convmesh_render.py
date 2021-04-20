import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def model(p):
    res = convMesh(p,
                   vect=(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
                         -1.0, 1.0, -1.0, 0.0, 1.0),
                   tri=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                   S=[7, 7, 7, 7],
                   T=5.2)

    return res


v, f, n = evalToMesh(model,
                     grid_min=(-2, -2, -2),
                     grid_max=(15, 15, 15),
                     grid_res=(64, 64, 64))

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
