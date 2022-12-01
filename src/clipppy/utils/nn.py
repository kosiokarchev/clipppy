from __future__ import annotations

from functools import partial
from typing import Callable, Type, Union

import torch
from torch import Size, Tensor
from torch.nn import (Conv1d, init, LazyLinear, Module, ReLU, Sequential, UninitializedBuffer)
from torch.nn.modules.conv import _ConvNd, LazyConv1d
from torch.nn.modules.lazy import LazyModuleMixin


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


class USequential(Sequential):
    def forward(self, arg):
        return arg, super().forward(arg)


def linear(size: int, nonlinearity: Union[Type[Module], Callable[[], Union[Module, Callable[[Tensor], Tensor]]]] = partial(ReLU, inplace=True), whiten=True):
    return Sequential(
        LazyLinear(size),
        *((LazyWhitenOnline(),) if whiten else ()),
        nonlinearity()
    )


def mlp(*sizes: int, nonlinearity: Union[Type[Module], Callable[[], Union[Module, Callable[[Tensor], Tensor]]]] = partial(ReLU, inplace=True), whiten=True):
    return Sequential(*map(partial(linear, nonlinearity=nonlinearity, whiten=whiten), sizes))


def omlp(*sizes: int, osize: int = 1, nonlinearity: Union[Type[Module], Callable[[], Union[Module, Callable[[Tensor], Tensor]]]] = partial(ReLU, inplace=True), whiten=True):
    return Sequential(mlp(*sizes, nonlinearity=nonlinearity, whiten=whiten), LazyLinear(osize))


class BatchedConv(_ConvNd):
    ndim: int

    @property
    def _ndim(self):
        return - (self.ndim + 2)

    def forward(self, input: Tensor) -> Tensor:
        res = super().forward(input.unsqueeze(self._ndim).flatten(end_dim=self._ndim))
        return res.reshape(input.shape[:self._ndim+1] + res.shape[self._ndim+1:])


class BatchedConv1d(BatchedConv, Conv1d):
    ndim = 1


class LazyBatchedConv1d(BatchedConv1d, LazyConv1d):
    cls_to_become = BatchedConv1d
    ndim = 1


class WhitenOnline(Module):
    ndim: int

    n: int
    mean: Tensor
    mean_square: Tensor
    std: Tensor

    _buffer_names = 'mean', 'mean_square', 'std'

    def __init__(self, shape=Size(), device=None, dtype=None):
        super().__init__()

        self.ndim = len(shape)
        self.n = 0

        for name in self._buffer_names:
            self.register_buffer(name, torch.empty(shape, device=device, dtype=dtype))
        self.reset_buffers()

    def reset_buffers(self):
        for name in self._buffer_names:
            init.zeros_(getattr(self, name))

    def forward(self, a: Tensor):
        if self.training:
            with torch.no_grad():
                n = a.shape[:a.ndim-self.ndim].numel()
                nnew = self.n + n
                wold, wnew = self.n/nnew, n/nnew
                _a = a.unsqueeze(0).flatten(end_dim=a.ndim-self.ndim)
                self.mean = _a.sum(0).div_(nnew).add_(self.mean, alpha=wold)
                self.mean_square = torch.linalg.vector_norm(_a, dim=0).pow_(2).div_(nnew).add_(self.mean_square, alpha=wold)
                self.std = (self.mean_square - self.mean**2).clamp_(1e-12).sqrt_()
                self.n = nnew

        return a if self.n < 2 else (a - self.mean).divide_(self.std)


class LazyWhitenOnline(LazyModuleMixin, WhitenOnline):
    cls_to_become = WhitenOnline

    mean: UninitializedBuffer
    mean_square: UninitializedBuffer
    std: UninitializedBuffer

    def __init__(self, ndim=1):
        super().__init__()
        self.ndim = ndim
        for name in self._buffer_names:
            setattr(self, name, UninitializedBuffer())

    def reset_buffers(self):
        if not self.has_uninitialized_params():
            super().reset_buffers()

    def initialize_parameters(self, a: Tensor):
        if self.has_uninitialized_params():
            with torch.no_grad():
                for name in self._buffer_names:
                    getattr(self, name).materialize(
                        shape=a.shape[a.ndim-self.ndim:],
                        device=a.device, dtype=a.dtype
                    )
                self.reset_buffers()
