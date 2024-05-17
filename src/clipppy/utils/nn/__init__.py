from __future__ import annotations

from functools import partial
from typing import Callable, Type, Union, Sequence, Iterable, Generic, Mapping, TypeVar

import attr
import torch
from frozendict import frozendict
from torch import Tensor, Size
from torch.nn import LazyLinear, Module, ReLU, Sequential, LayerNorm, ModuleDict

from .attrs import AttrsModule
from .whiten import LazyWhitenOnline, WhitenOnline
from ..typing import _ff_module_like
from .empty import _empty_module


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
Concat = partial(PostPartialModule, torch.concat)
Stack = partial(PostPartialModule, torch.stack)


@attr.s
class Mapper(AttrsModule):
    mod: _ff_module_like
    seq_cls: Type[Sequence[Tensor]] = tuple

    def forward(self, ts: Iterable[Tensor]):
        return self.seq_cls(map(self.mod, ts))


_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


@attr.s
class MultiModule(AttrsModule, Generic[_KT]):
    mods: Mapping[_KT, Module] = attr.ib(converter=ModuleDict)

    def forward(self, inputs: Mapping[_KT, _VT]):
        return type(inputs)((key, self.mods[key](val) if key in self.mods else val) for key, val in inputs.items())


class USequential(Sequential):
    def forward(self, arg):
        return arg, super().forward(arg)


class NLazyLinear(LazyLinear):
    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.size(-1)
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()


class Perceptron(Module):
    def __init__(
        self, size: int, nonlinearity: Union[Type[Module], Callable[[], _ff_module_like]] = partial(ReLU, inplace=True),
        whiten=False, layernorm=frozendict(elementwise_affine=False)
    ):
        super().__init__()
        self.lin = NLazyLinear(size)
        if whiten:
            self.norm = WhitenOnline(Size((size,)))
        elif layernorm:
            self.norm = LayerNorm(size, **({} if layernorm is True else layernorm))
        else:
            self.norm = _empty_module
        self.nonlin = nonlinearity()

    def forward(self, a):
        return self.nonlin(self.norm(self.lin(a)))


def mlp(*sizes: int, **kwargs):
    return Sequential(*map(partial(Perceptron, **kwargs), sizes))


def omlp(*sizes: int, osize: int = 1, **kwargs):
    return Sequential(mlp(*sizes, **kwargs), NLazyLinear(osize))
