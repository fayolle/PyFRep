import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.sdf_primitives import *
from FRep.sdf_ops import *
from FRep.ops import orient
from FRep.mesh import *

import polyscope as ps


def model(p):
    c1 = coneX(p, (0.0, 0.0, 0.0), 1.0)
    return c1


print('Generating simple model')
v, f, n = evalToMesh(model,
                     grid_min=(-2, -2, -2),
                     grid_max=(2, 2, 2),
                     grid_res=(64, 64, 64))
print('Done')

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
