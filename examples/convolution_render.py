import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def model(p):
    # res = convPoint(p, vect=(0.0,0.0,0.0,1.0,1.0,1.0,2.0,2.0,2.0), S=(2.0,2.0,2.0), T=0.1)
    # res = convLine(p, begin=(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0), end=(1.0,1.0,0.0,-1.0,-1.0,0.0,-1.0,1.0,0.0,1.0,-1.0,0.0), S=(3.0,3.0,3.0,3.0), T=0.5)
    '''
    res = convCurve(p,
                    vect=(1.0, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0,
                          1.0, 0.0, 1.0, 1.0, 0.0),
                    S=(2.0, 2.0, 2.0, 2.0),
                    T=0.1)
    '''
    '''
    res = ConvLineR (p,
                     begin=(0.0,0.0,0.0,1.0,1.0,0.0,1.0,-1.0,0.0,-1.0,-1.0,0.0),
                     end=(1.0,1.0,0.0,1.0,-1.0,0.0,-1.0,-1.0,0.0,-1.0,1.0,0.0),
                     S=0.6,
                     R=0.3)
    '''
    '''
    res = ConvTriangle(p,
                    vect=(0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0),
                    S=[5],
                    T=4.2)
    '''
    '''
    res = ConvMesh(p,
                       vect=(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                             0.0, 0.0, 0.0, -1.0, -1.0, 1.0, -1.0, 0.0, 1.0
                             ),
                       tri=(1,2,3,
                            4,5,6,
                            7,8,9,
                            10,11,12),
                       S=[7,7,7,7],
                       T=5.2)
    '''

    res = ConvArc(p,angle=[0.0],axis=[0,0,0],theta=[130],radius=[1],center = [0,0,0],
                   S=[0.1],
                   T=1.2)


    return res


v, f, n = evalToMesh(model,
                     grid_min=(-2, -2, -2),
                     grid_max=(15, 15, 15),
                     grid_res=(64, 64, 64))

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
