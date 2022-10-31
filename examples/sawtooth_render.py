'''
Replication of cells using the sawtooth function.
'''

import FRep
from FRep.primitives import *
from FRep.ops import *
from FRep.mesh import *

import polyscope as ps


def replicant(p):
    q = sawtooth(p, 5.0)
    torus1 = torusZ(q, (-1.0, 1.0, 0.0), 1.0, 0.25)
    torus2 = torusX(q, (0.0, -1.0, -1.0), 1.0, 0.25)
    torus3 = torusY(q, (1.0, 0.0, 1.0), 1.0, 0.25)
    sp4 = sphere(q, (0.0, 0.0, 0.0), 0.8)
    res5 = boundBlendUnion(torus1, torus2, sp4, 0.08, 1.0, 1.0, 1.0)
    res6 = boundBlendUnion(res5, torus3, sp4, 0.08, 1.0, 1.0, 1.0)
    return res6

v, f, n = evalToMesh(replicant,
                     grid_min=(-10.0, -10.0, -10.0),
                     grid_max=(10.0, 10.0, 10.0),
                     grid_res=(128, 128, 128))

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
