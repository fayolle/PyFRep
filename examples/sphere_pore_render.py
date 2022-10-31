'''
Replication of cells using the sawtooth function.
'''

import FRep
from FRep.primitives import *
from FRep.sdf_primitives import sphere as sdSphere
from FRep.ops import *
from FRep.sdf_ops import shell
from FRep.mesh import *

import polyscope as ps


def model(p):
    R = 10.0
    f_dist = sdSphere(p, center=(0.0,0.0,0.0), r=R)
    offset = 0.5
    f_shell = shell(f_dist, offset)

    dist1 = 0.0
    dist2 = 10.0
    t = (f_dist - dist1) / (dist2 - dist1)
    scale1 = 0.65
    scale2 = 0.95
    scale = (1.0 - t) * scale1 + t * scale2 + noiseG(p, 0.4, 0.9, 1.4)
    freq1 = 2.5
    freq2 = 1.1
    freq = (1 - t) * freq1 + t * freq2 + noiseG(p, 0.75, 1.2, 2.1)
    xt = torch.sin(p[:, 0] * freq) / scale
    yt = torch.sin(p[:, 1] * freq) / scale
    zt = torch.sin(p[:, 2] * freq) / scale
    pt = torch.zeros_like(p)
    pt[:, 0] = xt
    pt[:, 1] = yt
    pt[:, 2] = zt
    f_hole = sphere(pt, (0.0, 0.0, 0.0), 1.0) + noiseG(p, 0.4, 1.5, 1.0)

    temp = intersection(union(intersection(f_dist, -f_hole), f_shell),
                        -p[:, 1])
    return temp

v, f, n = evalToMesh(model,
                     grid_min=(-11.0, -11.0, -11.0),
                     grid_max=(11.0, 11.0, 11.0),
                     grid_res=(64, 64, 64))

ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
