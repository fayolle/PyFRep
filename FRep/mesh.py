import skimage.measure as sk
import numpy as np
import torch
from .grid import torchGrid, torchSampling


def evalToMesh(model, grid_min, grid_max, grid_res, device='cpu'):
    x, y, z = torchGrid(grid_min, grid_max, grid_res, device)
    volume = torchSampling(model, x, y, z)
    volume = volume.detach().cpu().numpy()

    vertices, faces, normals, _ = sk.marching_cubes(volume, level=0)

    return vertices, faces, normals

