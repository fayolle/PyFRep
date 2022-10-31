import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from skimage import measure


# MLP 
class Implicit(nn.Module):
    def __init__(self, dimension, radius=1):
        super(Implicit, self).__init__()

        d_in = dimension
        dims = [512, 512, 512, 512, 512, 512, 512, 512]
        beta = 100
        skip_in = [4]
        radius_init = radius
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
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                torch.nn.init.constant_(lin.bias, -radius_init)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            setattr(self, "lin" + str(layer), lin)
        self.activation = nn.Softplus(beta=beta)
    
    def forward(self, input):
        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
        return x


# MLP with a given implicit fun(x,y,z) used in the last layer 
class ImplicitFun(nn.Module):
    def __init__(self, fun, dimension, radius=1):
        super(ImplicitFun, self).__init__()

        d_in = dimension
        dims = [512, 512, 512, 512, 512, 512, 512, 512]
        beta = 100
        skip_in = [4]
        radius_init = radius
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
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                torch.nn.init.constant_(lin.bias, -radius_init)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            setattr(self, "lin" + str(layer), lin)
        
        self.activation = nn.Softplus(beta=beta)
        self.fun = fun

    def forward(self, input):
        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
        fun_sign = torch.tanh(0.1*self.fun(input))
        fun_sign = fun_sign.reshape(x.shape)
        x = x * fun_sign
        return x


# Utils 
def grad(y, x):
    g = autograd.grad(y, [x], grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return g


def div(y, x):
    div = 0.0
    for i in range(y.shape[-1]):
        div += autograd.grad(y[..., i], x, grad_outputs=torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def Laplacian(y, x):
    g = grad(y, x)
    return div(g, x)


def pLaplacian(y, x, p=2):
    g = grad(y, x)
    g_n = torch.linalg.norm(g, 2, dim=1)
    g_n = g_n**(p-2)
    g_n = torch.reshape(g_n, (g.shape[0],1))
    g = g_n * g
    return div(g, x)


def read_xyzn(filename, device):
    '''
    Read a 3d point-cloud (points with normals) and return a torch tensor 
    '''
    with open(filename, 'r') as f:
        raw_data = np.loadtxt(f)
    
    points, normals = np.hsplit(raw_data, 2)
  
    p = torch.from_numpy(points).float().to(device)

    norm = np.linalg.norm(normals, axis=0)
    unit_normals = normals/norm
    n = torch.from_numpy(unit_normals).float().to(device)

    data = torch.cat((p,n), dim=-1)

    data.requires_grad_()
  
    return data


def xyzn_bbox(xyzn):
    '''
    Compute the axis oriented bounding box (the corners): xmin and xmax 
    for a given 3d point-cloud (xyzn). 
    '''
    xmin = xyzn[:,0].min()
    xmax = xyzn[:,0].max()
    ymin = xyzn[:,1].min()
    ymax = xyzn[:,1].max()
    zmin = xyzn[:,2].min()
    zmax = xyzn[:,2].max()
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    d = 0.01 * torch.sqrt(dx*dx + dy*dy + dz*dz)

    xmin = xmin.cpu().detach().numpy()
    xmax = xmax.cpu().detach().numpy()
    ymin = ymin.cpu().detach().numpy()
    ymax = ymax.cpu().detach().numpy()
    zmin = zmin.cpu().detach().numpy()
    zmax = zmax.cpu().detach().numpy()
    d = d.cpu().detach().numpy()

    pmin = (xmin-d, ymin-d, zmin-d)
    pmax = (xmax+d, ymax+d, zmax+d)

    return (pmin, pmax)


def uniformSamples(num_points, grid_min, grid_max, device):
    xx = torch.FloatTensor(num_points, 1).uniform_(grid_min[0], grid_max[0])
    yy = torch.FloatTensor(num_points, 1).uniform_(grid_min[1], grid_max[1])
    zz = torch.FloatTensor(num_points, 1).uniform_(grid_min[2], grid_max[2])
    x = torch.cat((xx,yy,zz), dim=-1)
    return x.to(device)


def train(num_epochs, data, device, batch_size=None):
    '''
    Function for training an implicit model (MLP) to fit a given 3D point-cloud 
    '''

    # Weights for the loss terms 
    w_geo = 1.0
    w_grad = 0.1 # 1.0 

    dim = 3

    model = Implicit(dimension=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    if batch_size is None:
        batch_size = data.shape[0]

    for i in range(0, num_epochs):
        batches = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
        for batch in batches:
            point_batch = batch[:,0:dim]
            normal_batch = batch[:,dim:2*dim]

            # Change to train mode
            model.train()

            optimizer.zero_grad()

            # Compute loss
            # TotalLoss = sum_i f(x_i) + ||grad(f)(x_i) - n_i||^2

            # Forward pass
            f_data = model(point_batch)

            # sum |f(x_i)| at the points xi
            geo_loss = f_data.abs().mean()

            # sum ||n_i - \nabla f(xi)||^2
            normal_grad = grad(f_data, point_batch)
            grad_loss = (normal_grad - normal_batch).norm(2, dim=1).mean()

            loss = w_geo * geo_loss + w_grad * grad_loss

            loss.backward()
            optimizer.step()

    model.eval()
  
    return model


def train_eikonal(num_iters, fun, grid_min, grid_max, device):
    '''
    Function for training an implicit model (MLP) to fit 
    a given 3D point-cloud, while enforcing that the implicit behaves 
    like the signed distance to the surface. 
    '''

    assert(len(grid_min) == len(grid_max))
    dimension = len(grid_min)

    model = ImplicitFun(fun=fun, dimension=dimension).to(device)
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
        extra_constraint = torch.mean(torch.where(torch.abs(fun_d)<1e-3, torch.zeros_like(f_d), torch.exp(-1e2 * torch.abs(f_d))))

        loss = g_constraint + extra_constraint

        loss.backward()
        #clip_grad = 1.0
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

    model.eval()
  
    return model


def freeformFit(filename, num_epochs, batch_size, device):
    '''
    Return a function f(x,y,z) with zero level-set approximating 
    the point cloud corresponding to filename. 
    '''

    xyzn = read_xyzn(filename, device=device)
    #xmin, xmax = xyzn_bbox(xyzn)
    model_surf = train(num_epochs, xyzn, device=device, batch_size=batch_size)
    return model_surf, xyzn

 
def freeformDistFit(filename, num_iters, num_epochs, batch_size, device):
    '''
    Return a distance function d(x,y,z) with zero level-set approximating 
    the point cloud corresponding to filename 
    '''

    xyzn = read_xyzn(filename, device=device)
    xmin, xmax = xyzn_bbox(xyzn)
    model_surf = train(num_epochs, xyzn, device=device, batch_size=batch_size) 
    
    # Freeze the parameters (such that they don't get updated by train_eikonal)
    for param in model_surf.parameters():
        param.requires_grad = False

    # mean value of model_surf on the point-cloud 
    #f_xyzn = model_surf(xyzn[:,0:3])
    #c = f_xyzn.mean() 
    #c_no_grad = c.detach()
    #model_surf_offset = lambda x: model_surf(x) - c_no_grad 
    #model_eik = train_eikonal(num_iters=num_iters, fun=model_surf_offset, grid_min=xmin, grid_max=xmax, device=device)
    
    model_eik = train_eikonal(num_iters=num_iters, fun=model_surf, grid_min=xmin, grid_max=xmax, device=device)

    return model_eik, xyzn 

