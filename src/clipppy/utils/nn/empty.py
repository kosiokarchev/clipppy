from __future__ import annotations

from torch.nn import Module


class EmptyModule(Module):
    @staticmethod
    def forward(a): return a


_empty_module = EmptyModule()
