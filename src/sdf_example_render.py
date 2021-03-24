from sdf_primitives import *
from sdf_ops import *
from mesh import *

import polyscope as ps

def model(p):
    sp1 = sphere(p, r=1)
    b1 = box(p, b=1.5)
    t1 = intersection(sp1, b1)
    p1 = orient(p, np.array((1,0,0)))
    c1 = cylinder(p1, r=0.5)
    p2 = orient(p, np.array((0,1,0)))
    c2 = cylinder(p2, r=0.5)
    c3 = cylinder(p, r=0.5)
    t2 = difference(t1, c1)
    t3 = difference(t2, c2)
    t4 = difference(t3, c3)
    return t4

print('Generating simple model')
v,f,n = evalToMesh(model, grid_min=(-2,-2,-2), grid_max=(2,2,2), cell_size=0.04)
print('Done')

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()



