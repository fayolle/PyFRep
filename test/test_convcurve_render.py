import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def model(p):
    res = convCurve(p,
                    vect=(1.0, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0,
                          1.0, 0.0, 1.0, 1.0, 0.0),
                    S=(2.0, 2.0, 2.0, 2.0),
                    T=0.1)
    return res


v, f, n = evalToMesh(model,
                     grid_min=(-2, -2, -2),
                     grid_max=(4, 4, 4),
                     grid_res=(64, 64, 64))

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
