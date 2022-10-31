import torch
import torch.autograd as autograd


def grad(y, x):
    '''
    Given y = f(x), compute g = grad(f)(x). 
    Return g
    '''
    g = autograd.grad(y, [x],
                      grad_outputs=torch.ones_like(y),
                      create_graph=True)[0]
    g = torch.nan_to_num(g)
    return g


def funGrad(f, x):
    '''
    Given a function f, and a variable x, compute y = f(x) and g = grad(f)(x).
    Return (y, g)
    '''
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x.requires_grad = True
    y = f(x)
    g = grad(y, x)
    return (y, g)


def div(y, x):
    '''
    Given a vector field y = v(x), compute div y, 
    the divergence of the vector field. 
    '''
    div = 0.0
    for i in range(y.shape[-1]):
        div += autograd.grad(y[..., i],
                             x,
                             grad_outputs=torch.ones_like(y[..., i]),
                             create_graph=True)[0][..., i:i + 1]
    div = torch.nan_to_num(div)
    return div


def Laplacian(y, x):
    '''
    Given a function f, and a variable x, compute the Laplacian of f at x. 
    '''
    g = grad(y, x)
    return div(g, x)


def pLaplacian(y, x, p=2):
    '''
    Given a function f, and a variable x, 
    compute the p-Laplacian of f at x. 
    '''
    g = grad(y, x)
    g_n = torch.linalg.norm(g, 2, dim=1)
    g_n = g_n**(p - 2)
    g_n = torch.reshape(g_n, (g.shape[0], 1))
    g = g_n * g
    return div(g, x)

