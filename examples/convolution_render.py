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
                  center=[0.0, 0.0, 0.0],
                  radius=[2.0],
                  theta=[360.0],
                  axis=[0.0, 0.0, 1.0],
                  angle=[0.0],
                  S=[0.6],
                  T=0.5)

    return res


v, f, n = evalToMesh(model,
                     grid_min=(-10, -10, -10),
                     grid_max=(10, 10, 10),
                     grid_res=(64, 64, 64))


ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
