import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def model(p):
    b1 = block(p, vertex=(-4.0, -4.0, -4.0), dx=8.0, dy=8.0, dz=8.0)
    l2 = lidinoid(p)
    t3 = intersection(b1, l2)
    return t3


print('Generating simple model')
v, f, n = evalToMesh(model,
                     grid_min=(-5, -5, -5),
                     grid_max=(5, 5, 5),
                     grid_res=(64, 64, 64))
print('Done')

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
