import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def model(p):
    res = convArc(p,
                  center=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  radius=[8.5, 8.5, 8.5],
                  theta=[360.0, 360.0, 360.0],
                  axis=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                  angle=[0.0, 90.0, 90.0],
                  S=[0.65, 0.65, 0.65],
                  T=0.22)

    return res


v, f, n = evalToMesh(model,
                     grid_min=(-20, -20, -20),
                     grid_max=(20, 20, 20),
                     grid_res=(64, 64, 64))

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
