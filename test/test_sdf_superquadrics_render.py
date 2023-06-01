import FRep
from FRep.sdf_primitives import *
from FRep.sdf_ops import *
from FRep.mesh import *

import polyscope as ps


def model(p):
    sq1 = superQuadric(p, e=(1.0, 1.0), a=(1.0,1.0,1.0), r=(0.0,0.0,0.0), t=(-1.5,0.0,0.0))
    sq2 = superQuadric(p, e=(0.5, 0.5), a=(1.0,1.0,1.0), r=(0.0,0.75,0.0), t=(1.5,0.0,0.0))
    u3 = union(sq1, sq2)
    return u3


print('Generating simple model')
v, f, n = evalToMesh(model,
                     grid_min=(-4, -2, -2),
                     grid_max=(4, 2, 2),
                     grid_res=(64, 64, 64))
print('Done')

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
