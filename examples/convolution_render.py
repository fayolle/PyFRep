import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def model(p):

    # res = convPoint (p,vect=(0,0,0,1,1,1,2,2,2),S=(2,2,2),T=0.1)


    # res = convLine(p, begin=(0,0,0,0,0,0,0,0,0,0,0,0), end = (1,1,0,-1,-1,0,-1,1,0,1,-1,0), S=(3, 3, 3,3), T=0.5)

    res = convCurve (p,vect=(1,1,0,1,-1,0,-1,-1,0,-1,1,0,1,1,0),S=(2,2,2,2),T=0.1)



    return res

print('Generating simple model')
v,f,n = evalToMesh(model, grid_min=(-2,-2,-2), grid_max=(4,4,4), grid_res=(64,64,64))
print('Done')

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()



