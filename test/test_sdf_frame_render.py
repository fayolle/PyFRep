import FRep
from FRep.sdf_primitives import *
from FRep.mesh import *

import polyscope as ps


def model(p):
    f = frame(p, (1.0, 1.0, 1.0), 0.05)
    return f


print('Frame model')
v, f, n = evalToMesh(model,
                     grid_min=(-2, -2, -2),
                     grid_max=(2, 2, 2),
                     grid_res=(64, 64, 64))
print('Done')

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
