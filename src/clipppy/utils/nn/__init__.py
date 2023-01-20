from __future__ import annotations

from functools import partial
from typing import Callable, Type, Union, Sequence, Iterable

import attr
import torch
from torch import Tensor
from torch.nn import LazyLinear, Module, ReLU, Sequential

from .attrs import AttrsModule
from .whiten import LazyWhitenOnline
from ._typing import _ff_module_like


class PartialModule(Module):
    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _call_with_args(self, *args, kwargs):
        return self.func(*args, **{**self.kwargs, **kwargs})

    def forward(self, *args, **kwargs):
        return self._call_with_args(*self.args, *args, kwargs=kwargs)


class PostPartialModule(PartialModule):
    def forward(self, *args, **kwargs):
        return self._call_with_args(*args, *self.args, kwargs=kwargs)


Movedim = partial(PostPartialModule, torch.movedim)
Squeeze = partial(PostPartialModule, torch.squeeze)
Unsqueeze = partial(PostPartialModule, torch.unsqueeze)


@attr.s(auto_attribs=True, eq=False)
class Mapper(AttrsModule):
    mod: _ff_module_like
    seq_cls: Type[Sequence[Tensor]] = tuple

    def forward(self, ts: Iterable[Tensor]):
        return self.seq_cls(map(self.mod, ts))


class USequential(Sequential):
    def forward(self, arg):
        return arg, super().forward(arg)


def linear(size: int, nonlinearity: Union[Type[Module], Callable[[], _ff_module_like]] = partial(ReLU, inplace=True), whiten=True):
    return Sequential(
        LazyLinear(size),
        *((LazyWhitenOnline(),) if whiten else ()),
        nonlinearity()
    )


def mlp(*sizes: int, nonlinearity: Union[Type[Module], Callable[[], Union[Module, Callable[[Tensor], Tensor]]]] = partial(ReLU, inplace=True), whiten=True):
    return Sequential(*map(partial(linear, nonlinearity=nonlinearity, whiten=whiten), sizes))


def omlp(*sizes: int, osize: int = 1, nonlinearity: Union[Type[Module], Callable[[], Union[Module, Callable[[Tensor], Tensor]]]] = partial(ReLU, inplace=True), whiten=True):
    return Sequential(mlp(*sizes, nonlinearity=nonlinearity, whiten=whiten), LazyLinear(osize))
