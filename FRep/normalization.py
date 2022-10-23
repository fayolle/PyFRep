import torch
from .diff import grad


def normalize(f, x, method='Rvachev'):
    """Normalize the implicit surface f. 

    Normalize the implicit surface f() at the point x. 
    x is assumed to be a Torch tensor for which derivatives are tracked. 
    The default method for normalization is 'Rvachev'. Either 'Rvachev' 
    or 'Taubin' can be used as a normalization method. 
    """

    #if not torch.is_tensor(x):
    #    x = torch.tensor(x)
    #x.requires_grad = True

    if method == 'Taubin':
        return normalize_Taubin(f, x)
    else:
        return normalize_Rvachev(f, x)


def normalize_Rvachev(f, x):
    """Implements the (first order) Rvachev normalization.

    Implements the first order Rvachev normalization of the implicit surface 
    f() at the point x. 
    x is assumed to be a Torch tensor for which derivatives are tracked. 
    """

    fx = f(x)
    gradf = grad(fx, x)
    norm_gradf2 = gradf[:, 0]**2 + gradf[:, 1]**2
    w1 = fx / torch.sqrt(fx**2 + norm_gradf2)
    return w1


def normalize_Taubin(f, x):
    """Implements the Taubin normalization. 

    Implements the Taubin normalization of the implicit surface f() at the 
    point x. 
    x is assumed to be a Torch tensor for which derivatives are tracked. 
    """

    fx = f(x)
    gradf = grad(fx, x)
    norm_gradf2 = gradf[:, 0]**2 + gradf[:, 1]**2
    norm_gradf = torch.sqrt(norm_gradf2)
    d1 = fx / norm_gradf
    return d1
