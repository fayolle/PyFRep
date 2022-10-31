import torch

from .diff import grad, div


# Compute the mean curvature of f at v (v: points on the surface f=0)
def meanCurvature(f, v):
    if not torch.is_tensor(v):
        # array.copy() is to prevent the case
        # where the numpy array has a negative stride
        vt = torch.tensor(v.copy())
    else:
        vt = v

    vt.requires_grad = True

    # eval the implicit at the points
    ft = f(vt)

    # derivatives
    grad_ft = grad(ft, vt)
    norm_grad_ft = torch.linalg.norm(grad_ft, 2, dim=1)
    normalized_grad_ft = torch.zeros_like(grad_ft)
    normalized_grad_ft[:, 0] = grad_ft[:, 0] / norm_grad_ft
    normalized_grad_ft[:, 1] = grad_ft[:, 1] / norm_grad_ft
    normalized_grad_ft[:, 2] = grad_ft[:, 2] / norm_grad_ft
    mean_curvature = -0.5 * div(normalized_grad_ft, vt)

    mc = mean_curvature[:, 0].detach().cpu().numpy()
    return mc


def GaussianCurvature(f, v):
    if not torch.is_tensor(v):
        # array.copy() is to prevent the case
        # where the numpy array has a negative stride
        vt = torch.tensor(v.copy())
    else:
        vt = v

    vt.requires_grad = True

    # eval the implicit at the points
    ft = f(vt)

    # derivatives
    grad_ft = grad(ft, vt)
    norm_grad_ft = torch.linalg.norm(grad_ft, 2, dim=1)
    norm_grad_ft4 = norm_grad_ft**4

    grad_ft = grad_ft.reshape((grad_ft.shape[0], grad_ft.shape[1], 1))

    # Hessian
    H_ft = torch.zeros((v.shape[0], 3, 3))
    tmp0 = grad(grad_ft[:, 0], vt)
    H_ft[:, 0, 0] = tmp0[:, 0]
    H_ft[:, 0, 1] = tmp0[:, 1]
    H_ft[:, 0, 2] = tmp0[:, 2]
    tmp1 = grad(grad_ft[:, 1], vt)
    H_ft[:, 1, 0] = tmp1[:, 0]
    H_ft[:, 1, 1] = tmp1[:, 1]
    H_ft[:, 1, 2] = tmp1[:, 2]
    tmp2 = grad(grad_ft[:, 2], vt)
    H_ft[:, 2, 0] = tmp2[:, 0]
    H_ft[:, 2, 1] = tmp2[:, 1]
    H_ft[:, 2, 2] = tmp2[:, 2]

    # adjoint of Hessian
    Hstar_f = torch.zeros_like(H_ft)
    Hstar_f[:, 0,
            0] = H_ft[:, 1, 1] * H_ft[:, 2, 2] - H_ft[:, 1, 2] * H_ft[:, 2, 1]
    Hstar_f[:, 1,
            1] = H_ft[:, 0, 0] * H_ft[:, 2, 2] - H_ft[:, 0, 2] * H_ft[:, 2, 0]
    Hstar_f[:, 2,
            2] = H_ft[:, 0, 0] * H_ft[:, 1, 1] - H_ft[:, 0, 1] * H_ft[:, 1, 0]
    Hstar_f[:, 1,
            0] = H_ft[:, 0, 2] * H_ft[:, 2, 1] - H_ft[:, 0, 1] * H_ft[:, 2, 2]
    Hstar_f[:, 2,
            0] = H_ft[:, 0, 1] * H_ft[:, 1, 2] - H_ft[:, 0, 2] * H_ft[:, 1, 1]
    Hstar_f[:, 0,
            1] = H_ft[:, 1, 2] * H_ft[:, 2, 0] - H_ft[:, 1, 0] * H_ft[:, 2, 2]
    Hstar_f[:, 2,
            1] = H_ft[:, 1, 0] * H_ft[:, 0, 2] - H_ft[:, 0, 0] * H_ft[:, 1, 2]
    Hstar_f[:, 0,
            2] = H_ft[:, 1, 0] * H_ft[:, 2, 1] - H_ft[:, 1, 1] * H_ft[:, 2, 0]
    Hstar_f[:, 1,
            2] = H_ft[:, 0, 1] * H_ft[:, 2, 0] - H_ft[:, 0, 0] * H_ft[:, 2, 1]

    tmp1 = torch.matmul(Hstar_f, grad_ft)
    nv = grad_ft.shape[0]
    tmp2 = torch.bmm(grad_ft.view(nv, 1, 3), tmp1.view(nv, 3, 1))
    tmp3 = tmp2.reshape((nv))
    tmp4 = tmp3 / norm_grad_ft4

    Kg = tmp4.detach().cpu().numpy()
    return Kg


def principalCurvatures(f, v):
    '''
    Compute the principal curvatures kmin and kmax for the implicit surface f 
    at the points v. 
    '''
    H = meanCurvature(f, v)
    K = GaussianCurvature(f, v)
    h2k = torch.sqrt(H**2 - K)
    kmin = H - h2k
    kmax = H + h2k
    return (kmin, kmax)

