from primitives import *
from ops import *
from mesh import *

import polyscope as ps

def model(p):
    sp1 = sphere(p, center=np.array((0,0,0)), r=1)
    b1 = block(p, vertex=np.array((-0.75, -0.75, -0.75)), dx=1.5, dy=1.5, dz=1.5)
    t1 = intersection(sp1, b1)
    c1 = cylX(p, center=np.array((0,0,0)), r=0.5)
    c2 = cylY(p, center=np.array((0,0,0)), r=0.5)
    c3 = cylZ(p, center=np.array((0,0,0)), r=0.5)
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



