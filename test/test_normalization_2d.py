import math
import sys
import torch
import numpy as np

try:
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("This test file requires the optional Matplotlib module")
    print("Exiting ...")
    sys.exit(1)

import FRep
from FRep.normalization import normalize


# Helper functions
def ellipse(p):
    x = p[:, 0]
    y = p[:, 1]
    d = 1.0 - (x / 5.0)**2 - (y / 3.0)**2
    return d


# Create a 2D grid as a torch tensor
def torch_grid(xmin, xmax, ymin, ymax, resx=64, resy=64, device='cpu'):
    dx = xmax - xmin
    dy = ymax - ymin
    ed = 0.1*math.sqrt(dx*dx+dy*dy)
    x = torch.arange(xmin-ed, xmax+ed, step=(dx+2*ed)/float(resx))
    y = torch.arange(ymin-ed, ymax+ed, step=(dy+2*ed)/float(resy))
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    return xx.to(device), yy.to(device)


# Sample function f() on torch 2D grid x,y
def torch_sampling(f, x, y, device='cpu'):
    nx = x.shape[0]
    ny = x.shape[1]
    d = nx * ny
    xy = torch.stack((x, y), dim=-1).reshape(d, 2)
    z = f(xy)
    z = torch.reshape(z, (nx, ny))
    return z


def show_contour_plot(x, y, f):
    xx = x.detach().numpy()
    yy = y.detach().numpy()
    ff = f.detach().numpy()
    plt.figure(figsize=(8, 4))
    h = plt.contourf(xx, yy, ff)
    h.ax.axis('equal')
    plt.title('Filled Contour Plot')
    plt.show()


def show_contour_lines(x, y, f):
    xx = x.detach().numpy()
    yy = y.detach().numpy()
    ff = f.detach().numpy()
    plt.figure()
    levels = np.arange(-0.5, 0.1, 0.1)
    CS = plt.contour(xx, yy, ff, levels)
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.title('Ellipse')
    plt.axis('equal')
    plt.show()


# Grid parameters
NX = 64
NY = 64
XMIN = 0.0
XMAX = 7.0
YMIN = -4.0
YMAX = 4.0

x, y = torch_grid(XMIN, XMAX, YMIN, YMAX, NX, NY)
f = torch_sampling(ellipse, x, y)
show_contour_lines(x, y, f)

x, y = torch_grid(XMIN, XMAX, YMIN, YMAX, NX, NY)
x.requires_grad = True
y.requires_grad = True


def w1(p): return normalize(ellipse, p, method='Rvachev')


f = torch_sampling(w1, x, y)
show_contour_lines(x, y, f)


def d1(p): return normalize(ellipse, p, method='Taubin')


f = torch_sampling(d1, x, y)
show_contour_lines(x, y, f)
