import FRep
from FRep.sdf_primitives import *
from FRep.sdf_ops import *
from FRep.ops import orient
from FRep.mesh import *

import polyscope as ps


def model(p):
    sp1 = sphere(p, center=(0.0,0.0,0.0), r=1)
    b1 = box(p, b=1.5)
    t1 = intersection(sp1, b1)
    c1 = cylinder(p, center=(0.0,0.0,0.0), u=(1.0,0.0,0.0), r=0.5)
    c2 = cylinder(p, center=(0.0,0.0,0.0), u=(0.0,1.0,0.0), r=0.5)
    c3 = cylinder(p, center=(0.0,0.0,0.0), u=(0.0,0.0,1.0), r=0.5)
    t2 = difference(t1, c1)
    t3 = difference(t2, c2)
    t4 = difference(t3, c3)
    return t4

v, f, n = evalToMesh(model,
                     grid_min=(-2, -2, -2),
                     grid_max=(2, 2, 2),
                     grid_res=(64, 64, 64))

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
