import torch
import torch.autograd as autograd


# Given y = f(x), compute grad(f)(x)
def grad(y, x):
    g = autograd.grad(y, [x], grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return g

# Given a function f, and a variable x,
# compute y = f(x) and g = grad(f)(x).
# Return (y, g)
def fun_grad(f, x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x.requires_grad = True
    y = f(x)
    g = grad(y, x)
    return (y, g)

# Given a vector field y = v(x), compute div y
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
    #g_n = g.norm(2, dim=1)
    g_n = torch.linalg.norm(g, 2, dim=1)
    g_n = g_n**(p-2)
    g_n = torch.reshape(g_n, (g.shape[0],1))
    g = g_n * g
    return div(g, x)


'''
# Example of use
def f(x):
  return x[:,0]**2 + x[:,1]**2 - 1 

x = torch.tensor([[2.0,2.0]], requires_grad=True)
y = f(x)
z = Laplacian(y, x)
print(y)
print(z)
'''
