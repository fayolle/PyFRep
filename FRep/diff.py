import torch
import torch.autograd as autograd

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

'''
# Example of use
def f(x):
  return x[0,0]**2 + x[0,1]**2 - 1 

x = torch.tensor([[2.0,2.0]], requires_grad=True)
y = f(x)
z = Laplacian(y, x)
print(y)
print(z)
'''
