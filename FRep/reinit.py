'''
Re-initialization of an implicit
'''

import torch
import torch.nn as nn
import numpy as np

from .diff import grad, Laplacian, pLaplacian


# Implicit as a neural network
class ImplicitNNFun(nn.Module):
    def __init__(self, fun, dimension):
        super(ImplicitNNFun, self).__init__()

        d_in = dimension
        dims = [512, 512, 512, 512, 512, 512, 512, 512]
        beta = 100
        skip_in = [4]
        radius_init = 1
        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)
            if layer == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight,
                                      mean=np.sqrt(np.pi) /
                                      np.sqrt(dims[layer]),
                                      std=0.00001)
                torch.nn.init.constant_(lin.bias, -radius_init)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0,
                                      np.sqrt(2) / np.sqrt(out_dim))
            setattr(self, "lin" + str(layer), lin)
        self.activation = nn.Softplus(beta=beta)
        #self.activation = nn.ReLU()
        self.fun = fun

    def forward(self, input):
        # to take deriv. wrt input
        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
        fun_sign = torch.tanh(0.1 * self.fun(input))
        fun_sign = fun_sign.reshape(x.shape)
        x = x * fun_sign
        return x


def loadModel(path, dim=3, device='cpu'):
    model = ImplicitNNFun(dim).to(device)
    try:
        checkpoint = torch.load(path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print('Failing to load a model. Creating a new one.')
    return model


def saveModel(path, model):
    torch.save({'model_state_dict': model.state_dict()}, path)


# Assume that we work in 3D, see below for a generic function
def uniformSamples(num_points, grid_min, grid_max, device):
    xx = torch.FloatTensor(num_points, 1).uniform_(grid_min[0], grid_max[0])
    yy = torch.FloatTensor(num_points, 1).uniform_(grid_min[1], grid_max[1])
    zz = torch.FloatTensor(num_points, 1).uniform_(grid_min[2], grid_max[2])
    x = torch.cat((xx, yy, zz), dim=-1)
    return x.to(device)


def uniformSamples_generic(num_points, grid_min, grid_max, device):
    n = len(grid_min)
    uu = []
    for i in range(n):
        u = torch.FloatTensor(num_points, 1).uniform_(grid_min[i], grid_max[i])
        uu.append(u)
    x = torch.cat(uu, dim=-1)
    return x.to(device)


def trainPPoisson(num_iters, fun, grid_min, grid_max, p=2, device='cpu'):
    assert (len(grid_min) == len(grid_max))
    dimension = len(grid_min)

    model = ImplicitNNFun(fun=fun, dimension=dimension).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    n_samples = 1024
    loss = 0

    # Train network
    for i in range(0, num_iters):
        # Uniform samples
        x = uniformSamples(n_samples, grid_min, grid_max, device)
        x.requires_grad = True

        # Input implicit
        fun_d = fun(x)

        model.train()
        optimizer.zero_grad()

        f_d = model(x)

        # p-Laplacian
        lap = pLaplacian(f_d, x, p)
        lap_constraint = torch.mean((lap + 1)**2)

        # Penalize extra zeros
        extra_constraint = torch.mean(
            torch.where(
                torch.abs(fun_d) < 1e-3, torch.zeros_like(f_d),
                torch.exp(-1e2 * torch.abs(f_d))))

        loss = lap_constraint + extra_constraint

        loss.backward()

        #clip_grad = 1.0
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()

    # model.eval()

    return model


def trainEikonal(num_iters, fun, grid_min, grid_max, device='cpu'):
    assert (len(grid_min) == len(grid_max))
    dimension = len(grid_min)

    model = ImplicitNNFun(fun=fun, dimension=dimension).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    n_samples = 1024
    loss = 0

    # Train network
    for i in range(0, num_iters):
        # Uniform samples
        x = uniformSamples(n_samples, grid_min, grid_max, device)
        x.requires_grad = True

        # Input implicit
        fun_d = fun(x)

        model.train()
        optimizer.zero_grad()

        f_d = model(x)

        g_d = grad(f_d, x)
        g_norm = (g_d.norm(2, dim=1) - 1)**2
        g_constraint = torch.mean(g_norm)

        # Penalize extra zeros
        extra_constraint = torch.mean(
            torch.where(
                torch.abs(fun_d) < 1e-3, torch.zeros_like(f_d),
                torch.exp(-1e2 * torch.abs(f_d))))

        loss = g_constraint + extra_constraint

        loss.backward()

        #clip_grad = 1.0
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()

    # model.eval()

    return model
