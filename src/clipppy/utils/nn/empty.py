from __future__ import annotations

from torch.nn import Module


class EmptyModule(Module):
    # noinspection PyUnusedLocal
    @staticmethod
    def forward(a, *args, **kwargs): return a


_empty_module = EmptyModule()
