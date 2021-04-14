import skimage.measure as sk
import numpy as np
import torch
from .grid import torchGrid, torchSampling


# Vertices returned by marching_cubes are in [0,M]x[0,N]x[0,P]
# where [M,N,P] are the dimensions of the volume grid. 
# They should correspond instead to the geometric coordinates of the object
def evalToMesh(model, grid_min, grid_max, grid_res, device='cpu'):
    x, y, z = torchGrid(grid_min, grid_max, grid_res, device)
    volume = torchSampling(model, x, y, z)
    volume = volume.detach().cpu().numpy()

    # Vertices returned by marching_cubes are in [0,M]x[0,N]x[0,P]
    # where [M,N,P] are the dimensions of the volume grid. 
    # They should correspond instead to the geometric coordinates of the object
    vertices, faces, normals, _ = sk.marching_cubes(volume, level=0)

    # vertices in [0,1]
    vertices[:,0] = vertices[:,0] / grid_res[0]
    vertices[:,1] = vertices[:,1] / grid_res[1]
    vertices[:,2] = vertices[:,2] / grid_res[2]
    
    # in [0, max-min] for each direction
    vertices[:,0] = vertices[:,0] * (grid_max[0]-grid_min[0])
    vertices[:,1] = vertices[:,1] * (grid_max[1]-grid_min[1])
    vertices[:,2] = vertices[:,2] * (grid_max[2]-grid_min[2])
    
    # in [min, max] for each direction
    vertices[:,0] = vertices[:,0] + grid_min[0]
    vertices[:,1] = vertices[:,1] + grid_min[1]
    vertices[:,2] = vertices[:,2] + grid_min[2]

    return vertices, faces, normals

