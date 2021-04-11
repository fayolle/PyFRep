import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *


def model(p):
    sp1 = sphere(p, center=(0,0,0), r=1)
    b1 = block(p, vertex=(-0.75, -0.75, -0.75), dx=1.5, dy=1.5, dz=1.5)
    t1 = intersection(sp1, b1)
    c1 = cylX(p, center=(0,0,0), r=0.5)
    c2 = cylY(p, center=(0,0,0), r=0.5)
    c3 = cylZ(p, center=(0,0,0), r=0.5)
    t2 = difference(t1, c1)
    t3 = difference(t2, c2)
    t4 = difference(t3, c3)
    return t4

print('Generating simple model')
writeMesh('simple.off', model, grid_min=(-2,-2,-2), grid_max=(2,2,2), grid_res=(64,64,64))
print('Done')


