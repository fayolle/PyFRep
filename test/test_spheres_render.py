import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def model(p):
    sp1 = sphere(p, center=(0, 0, 0), r=1)
    sp2 = sphere(p, center=(0, 1, 0), r=1)
    t3 = union(sp1, sp2)
    return t3


print('Generating simple model')
v, f, n = evalToMesh(model,
                     grid_min=(-2, -2, -2),
                     grid_max=(3, 3, 3),
                     grid_res=(64, 64, 64))
print('Done')

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
