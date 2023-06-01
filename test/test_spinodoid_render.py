import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def model(p):
    sp = spinodoid(p, wavenumber=15.0*3.14159, numwaves=1000, density=0.5)
    bb = block(p, vertex=(0.0,0.0,0.0), dx=1.0, dy=1.0, dz=1.0)
    t = intersection(sp, bb)
    return t


print('Generating simple model')
v, f, n = evalToMesh(model,
                     grid_min=(-0.5, -0.5, -0.5),
                     grid_max=(1.5, 1.5, 1.5),
                     grid_res=(64, 64, 64))
print('Done')

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
