'''
Example to illustrate computing the mean curvature of an implicit surface 
and rendering it. 
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch

import FRep
from FRep.primitives import sphere
from FRep.mesh import evalToMesh
from FRep.curvature import meanCurvature, GaussianCurvature
from FRep.IO import writeOFF

import polyscope as ps


def model(p):
    sp1 = sphere(p, center=(0,0,0), r=1)
    return sp1

# Generate the mesh
v,f,n = evalToMesh(model, grid_min=(-2,-2,-2), grid_max=(2,2,2), grid_res=(64,64,64))

# Evaluate the mean curvature at the vertices
Km = meanCurvature(model, v)
Kg = GaussianCurvature(model, v)

ps.init()
ps.register_surface_mesh("mesh", v, f, smooth_shade=True)
ps.get_surface_mesh("mesh").add_scalar_quantity("mean curvature", Km, defined_on='vertices', cmap='jet', enabled=True)
ps.get_surface_mesh("mesh").add_scalar_quantity("Gaussian curvature", Kg, defined_on='vertices', cmap='jet', enabled=False)
ps.show()

