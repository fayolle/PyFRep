'''
Replication of cells using the sawtooth function.
'''

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def replicant(p):
    q = sawtooth(p, 5.0)
    torus1 = torusZ(q, (-1.0, 1.0, 0.0), 1.0, 0.1)
    torus2 = torusX(q, (0.0, -1.0, -1.0), 1.0, 0.1)
    torus3 = torusY(q, (1.0, 0.0, 1.0), 1.0, 0.1)
    res4 = union(torus1, torus2)
    res5 = union(res4, torus3)
    return res5


print('Generating simple model')
v, f, n = evalToMesh(replicant,
                     grid_min=(-10.0, -10.0, -10.0),
                     grid_max=(10.0, 10.0, 10.0),
                     grid_res=(64, 64, 64))
print('Done')

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
