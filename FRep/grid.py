import numpy as np
import torch


def npGrid(grid_min, grid_max, grid_res):
    nx, ny, nz = grid_res
    x_ = np.linspace(grid_min[0], grid_max[0], nx)
    y_ = np.linspace(grid_min[1], grid_max[1], ny)
    z_ = np.linspace(grid_min[2], grid_max[2], nz)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    return x, y, z


def npLinearGrid(grid_min, grid_max, grid_res):
    nx, ny, nz = grid_res
    x_ = np.linspace(grid_min[0], grid_max[0], nx)
    y_ = np.linspace(grid_min[1], grid_max[1], ny)
    z_ = np.linspace(grid_min[2], grid_max[2], nz)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    xyz = np.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), axis=1)
    return xyz


def npSampling(model, x, y, z):
    nx = x.shape[0]
    ny = y.shape[0]
    nz = z.shape[0]
    p = np.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), axis=1)
    d = model(p)
    #volume = d.reshape((nx,ny,nz)).transpose()
    volume = d.reshape((nx, ny, nz))
    return volume


def npLinearSampling(model, xyz):
    d = model(xyz)
    return d


def torchGrid(grid_min, grid_max, grid_res, device='cpu'):
    resx, resy, resz = grid_res
    dx = grid_max[0] - grid_min[0]
    x = torch.arange(grid_min[0], grid_max[0], step=dx / float(resx))
    dy = grid_max[1] - grid_min[1]
    y = torch.arange(grid_min[1], grid_max[1], step=dy / float(resy))
    dz = grid_max[2] - grid_min[2]
    z = torch.arange(grid_min[2], grid_max[2], step=dz / float(resz))
    xx, yy, zz = torch.meshgrid(x, y, z)
    return xx.to(device), yy.to(device), zz.to(device)


def torchSampling(model, x, y, z):
    # with torch.no_grad():
    # Evaluate function on each grid point
    resx = x.shape[0]
    resy = y.shape[1]
    resz = z.shape[2]
    dimg = resx * resy * resz
    p = torch.stack((x, y, z), dim=-1).reshape(dimg, 3)
    d = model(p)
    volume = torch.reshape(d, (resx, resy, resz))
    return volume


def torchLinearGrid(grid_min, grid_max, grid_res, device='cpu'):
    resx, resy, resz = grid_res
    dx = grid_max[0] - grid_min[0]
    x = torch.arange(grid_min[0], grid_max[0], step=dx / float(resx))
    dy = grid_max[1] - grid_min[1]
    y = torch.arange(grid_min[1], grid_max[1], step=dy / float(resy))
    dz = grid_max[2] - grid_min[2]
    z = torch.arange(grid_min[2], grid_max[2], step=dz / float(resz))
    xx, yy, zz = torch.meshgrid(x, y, z)
    xx = xx.to(device)
    yy = yy.to(device)
    zz = zz.to(device)
    dimg = resx * resy * resz
    xyz = torch.stack((xx, yy, zz), dim=-1).reshape(dimg, 3)
    return xyz


def torchLinearSampling(model, xyz):
    d = model(xyz)
    return d


# Return a uniform grid 
# grid_min, grid_max: arrays of dim 3
# grid_res: integer (resolution along each axis)
# device: device type (cuda or cpu) for torch 
# indexing: string specifying the type of indexing to use 
def getUniformGrid(grid_min, grid_max, grid_res, device, indexing = 'mc'):
    eps = 0.1
    
    bounding_box = grid_max - grid_min
    shortest_axis = np.argmin(bounding_box)

    if (shortest_axis == 0):
        x = np.linspace(grid_min[shortest_axis] - eps, grid_max[shortest_axis] + eps, grid_res)
        length = np.max(x) - np.min(x)
        y = np.arange(grid_min[1] - eps, grid_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(grid_min[2] - eps, grid_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(grid_min[shortest_axis] - eps, grid_max[shortest_axis] + eps, grid_res)
        length = np.max(y) - np.min(y)
        x = np.arange(grid_min[0] - eps, grid_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(grid_min[2] - eps, grid_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(grid_min[shortest_axis] - eps, grid_max[shortest_axis] + eps, grid_res)
        length = np.max(z) - np.min(z)
        x = np.arange(grid_min[0] - eps, grid_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(grid_min[1] - eps, grid_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    # Different types of indexing for meshing with the Marching Cubes algorithm 
    # or exporting to VTK 
    if (indexing == 'mc'):
        xx, yy, zz = np.meshgrid(x, y, z)
    elif (indexing == 'vtk'):
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    if device.type == 'cuda': 
        grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    else:
        grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cpu()

    return {"grid_points":grid_points, "xyz":[x,y,z], "resolution": [xx.shape[0], xx.shape[1], xx.shape[2]]}


def genVoxels(model, grid_min, grid_max, grid_res, device):
    with torch.no_grad():
        model.eval()

        # volumetric grid for sampling the model
        grid = getUniformGrid(grid_min, grid_max, grid_res, device, indexing='vtk')

        z = []

        for i,pnts in enumerate(torch.split(grid['grid_points'], 100000, dim=0)):
            z.append(model(pnts).detach().cpu().numpy())

        z = np.concatenate(z,axis=0)

        # Return the (linearized) grid
        return grid, z

