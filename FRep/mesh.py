import skimage.measure as sk
import numpy as np
import torch
from .grid import torchGrid, torchSampling, getUniformGrid 


def evalToMesh(model, grid_min, grid_max, grid_res, device='cpu'):
    x, y, z = torchGrid(grid_min, grid_max, grid_res, device)
    volume = torchSampling(model, x, y, z)
    volume = volume.detach().cpu().numpy()

    # Vertices returned by marching_cubes are in [0,M]x[0,N]x[0,P]
    # where [M,N,P] are the dimensions of the volume grid.
    # They should correspond instead to the geometric coordinates of the object
    vertices, faces, normals, _ = sk.marching_cubes(volume, level=0)

    # vertices in [0,1]
    vertices[:, 0] = vertices[:, 0] / grid_res[0]
    vertices[:, 1] = vertices[:, 1] / grid_res[1]
    vertices[:, 2] = vertices[:, 2] / grid_res[2]

    # in [0, max-min] for each direction
    vertices[:, 0] = vertices[:, 0] * (grid_max[0] - grid_min[0])
    vertices[:, 1] = vertices[:, 1] * (grid_max[1] - grid_min[1])
    vertices[:, 2] = vertices[:, 2] * (grid_max[2] - grid_min[2])

    # in [min, max] for each direction
    vertices[:, 0] = vertices[:, 0] + grid_min[0]
    vertices[:, 1] = vertices[:, 1] + grid_min[1]
    vertices[:, 2] = vertices[:, 2] + grid_min[2]

    return vertices, faces, normals


# Return a triangle mesh:
# (verts, faces)
# where verts is the list of vertex coordinates
# and faces is the list of polygon indices.
# 
# Note: grid_res is a scalar here 
# (assume the grid resolution is grid_res x grid_res x grid_res)
#
# Ultimately this should replace evalToMesh()
def evalToMesh2(model, grid_min, grid_max, grid_res, device, mc_value = 0.0):
    with torch.no_grad():
        model.eval()

        # volumetric grid for sampling the model
        grid = getUniformGrid(grid_min, grid_max, grid_res, device, indexing='mc')

        z = []

        for i,pnts in enumerate(torch.split(grid['grid_points'], 100000, dim=0)):
            z.append(model(pnts).detach().cpu().numpy())

        z = np.concatenate(z,axis=0)

        if (not (np.min(z) > mc_value or np.max(z) < mc_value)):
            z  = z.astype(np.float64)

            #verts, faces, normals, values = measure.marching_cubes_lewiner(
            verts, faces, normals, values = measure.marching_cubes(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0], grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=mc_value,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1]))

            verts = verts + np.array([grid['xyz'][0][0],grid['xyz'][1][0],grid['xyz'][2][0]])

        # Return the mesh
        return verts, faces

