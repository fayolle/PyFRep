import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def model(p):
    b1 = block(p, vertex=np.array((-2.0, -2.0, -2.0)), dx=4.0, dy=4.0, dz=4.0)
    g2 = gyroid_sheet(p, alpha=0.5, beta=0.5, gamma=0.5, c1=0.8, c2=-0.8)
    t3 = intersection(b1, g2)
    return t3

print('Generating simple model')
v,f,n = evalToMesh(model, grid_min=(-3,-3,-3), grid_max=(3,3,3), cell_size=0.06)
print('Done')

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()



