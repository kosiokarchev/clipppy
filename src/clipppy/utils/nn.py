from __future__ import annotations

from functools import partial

import torch
from torch import Tensor
from torch.nn import Module


class EmptyModule(Module):
    @staticmethod
    def forward(a): return a


_empty_module = EmptyModule()


class PartialModule(Module):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = partial(func, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


Movedim = partial(PartialModule, torch.movedim)
Squeeze = partial(PartialModule, torch.squeeze)
Unsqueeze = partial(PartialModule, torch.unsqueeze)


class WhitenOnline(Module):
    def __init__(self, ndim=1):
        super().__init__()
        self.ndim = ndim

        self.n = 0
        self.mean = self.mean_square = self.std = 0.

    def forward(self, a: Tensor):
        if self.training:
            with torch.no_grad():
                n = a.shape[:a.ndim-self.ndim].numel()
                nnew = self.n + n
                wold, wnew = self.n/nnew, n/nnew
                self.mean, self.mean_square = (
                    wold * old + new.unsqueeze(0).flatten(end_dim=new.ndim-self.ndim).sum(0) / nnew
                    for old, new in ((self.mean, a), (self.mean_square, a**2))
                )
                self.std = (self.mean_square - self.mean**2).clamp_(1e-12)**0.5
                self.n = nnew

        return a if self.n < 2 else (a - self.mean).divide_(self.std)
