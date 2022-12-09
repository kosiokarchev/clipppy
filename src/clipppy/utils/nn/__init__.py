from __future__ import annotations

from functools import partial
from typing import Callable, Type, Union

import torch
from torch import Tensor
from torch.nn import LazyLinear, Module, ReLU, Sequential

from .whiten import LazyWhitenOnline


class PartialModule(Module):
    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.func(*self.args, *args, **{**self.kwargs, **kwargs})


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
