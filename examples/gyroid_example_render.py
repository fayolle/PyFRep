import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def model(p):
    b1 = block(p, vertex=(-2.0, -2.0, -2.0), dx=4.0, dy=4.0, dz=4.0)
    g2 = gyroid(p, alpha=0.5, beta=0.5, gamma=0.5)
    t3 = intersection(b1, g2)
    return t3

v, f, n = evalToMesh(model,
                     grid_min=(-3, -3, -3),
                     grid_max=(3, 3, 3),
                     grid_res=(64, 64, 64))

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
