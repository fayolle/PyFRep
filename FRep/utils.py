#import numpy as np
import torch

_min = torch.minimum
_max = torch.maximum


def _normalize(a):
    return a / torch.linalg.norm(a)


def _vec(*arrs):
    return torch.stack(arrs, axis=-1)


def _length(a):
    return torch.linalg.norm(a, axis=1)


def _dot(a, b):
    return torch.sum(a * b, axis=1)
