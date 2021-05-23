import torch

import FRep
from FRep.primitives import sphere
from FRep.diff import grad

x = torch.tensor([[1.0, 0.0, 0.0]])
x.requires_grad = False
center = torch.tensor([0.0, 0.0, 0.0])
center.requires_grad = False
r = torch.tensor(1.0)
r.requires_grad = True

f = sphere(x, center, r)
dfdr = grad(f, r)
print(dfdr)
