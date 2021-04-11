import numpy as np
import torch


def npGrid(grid_min, grid_max, grid_res):
    nx, ny, nz = grid_res
    x_ = np.linspace(grid_min[0], grid_max[0], nx)
    y_ = np.linspace(grid_min[1], grid_max[1], ny)
    z_ = np.linspace(grid_min[2], grid_max[2], nz)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    return x, y, z

def npSampling(model, x, y, z):
    nx = x.shape[0]
    ny = y.shape[0]
    nz = z.shape[0]
    p = np.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), axis=1)
    d = model(p)
    volume = d.reshape((nx,ny,nz)).transpose()
    return volume

def torchGrid(grid_min, grid_max, grid_res, device='cpu'):
    resx, resy, resz = grid_res
    dx = grid_max[0]-grid_min[0]
    x = torch.arange(grid_min[0], grid_max[0], step=dx/float(resx))
    dy = grid_max[1]-grid_min[1]
    y = torch.arange(grid_min[1], grid_max[1], step=dy/float(resy))
    dz = grid_max[2]-grid_min[2]
    z = torch.arange(grid_min[2], grid_max[2], step=dz/float(resz))
    xx, yy, zz = torch.meshgrid(x, y, z)
    return xx.to(device), yy.to(device), zz.to(device)

def torchSampling(model, x, y, z):
    # with torch.no_grad():
    # Evaluate function on each grid point
    resx = x.shape[0]
    resy = y.shape[1]
    resz = z.shape[2]
    dimg = resx * resy * resz
    p = torch.stack((xx, yy, zz), dim=-1).reshape(dimg,3)
    d = model(p)
    volume = torch.reshape(z, (resx,resy, resz))
    return volume
