from __future__ import annotations

import os
import pickle
from functools import partial
from typing import Callable, Type, Union, Sequence, Iterable, Generic, Mapping, Optional

import attr
import torch
from frozendict import frozendict
from torch import Tensor, Size
from torch.nn import LazyLinear, Module, ReLU, Sequential, LayerNorm, ModuleDict
from torch.nn.modules.lazy import LazyModuleMixin

from phytorchx.attrs import AttrsModule
from .empty import _empty_module
from .whiten import LazyWhitenOnline, WhitenOnline, BatchedSyncBatchNorm, LazyBatchedSyncBatchNorm
from ..typing import _ff_module_like, _KT, _VT


class BSPickler(pickle.Pickler):
    def __init__(self):
        super().__init__(open(os.devnull, 'wb'))
        self.refs = set()

    def reducer_override(self, obj):
        if isinstance(obj, torch.Tensor):
            self.refs.add(obj)
            return object, ()
        else:
            return NotImplemented

    def __call__(self, obj):
        self.dump(obj)
        return self.refs


def extract_extra_buffers(o: Module):
    return BSPickler()(o) - set(o.parameters()) - set(o.buffers())


def extract_buffers(t, base: Type):
    if isinstance(t, base):
        t = t.__dict__.values()

    if isinstance(t, torch.Tensor):
        yield t
    elif isinstance(t, Iterable):
        for v in t:
            yield from extract_buffers(v, base)


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


@attr.s(eq=False, auto_attribs=True)
class Mapper(AttrsModule):
    mod: _ff_module_like
    seq_cls: Type[Sequence[Tensor]] = tuple

    def forward(self, ts: Iterable[Tensor]):
        return self.seq_cls(map(self.mod, ts))


@attr.s(eq=False, auto_attribs=True)
class MultiModule(AttrsModule, Generic[_KT]):
    mods: Mapping[_KT, Module] = attr.ib(converter=ModuleDict)

    def forward(self, inputs: Mapping[_KT, _VT]):
        return type(inputs)((key, self.mods[key](val) if key in self.mods else val) for key, val in inputs.items())


class USequential(Sequential):
    def forward(self, arg):
        return arg, super().forward(arg)


class SquareLazyLinear(LazyLinear):
    def __init__(self, bias: bool = True, device=None, dtype=None):
        super().__init__(0, bias, device, dtype)

    def initialize_parameters(self, input):
        self.out_features = input.size(-1)
        return super().initialize_parameters(input)


def Norm(size=None, whiten=False, batchnorm=False, layernorm=False):
    for arg, cls, lazy_cls in (
        (whiten, lambda s, **kwargs: WhitenOnline(Size((s,)), **kwargs), LazyWhitenOnline),
        (batchnorm, BatchedSyncBatchNorm, LazyBatchedSyncBatchNorm),
        (layernorm, LayerNorm, None)
    ):
        if arg:
            return (lazy_cls if size is None else partial(cls, size))(**({} if arg is True else arg))
    return _empty_module


class Perceptron(Module):
    def __init__(
        self, size: Optional[int], nonlinearity: Union[Type[Module], Callable[[], _ff_module_like]] = ReLU,
        bias=True, whiten=False, batchnorm=False, layernorm=frozendict(elementwise_affine=False)
    ):
        super().__init__()
        self.lin = LazyLinear(size, bias=bias)
        self.norm = Norm(size, whiten, batchnorm, layernorm)
        self.nonlin = nonlinearity()

    def forward(self, a):
        return self.nonlin(self.norm(self.lin(a)))


def mlp(*sizes: int, **kwargs):
    return Sequential(*map(partial(Perceptron, **kwargs), sizes))


def omlp(*sizes: int, osize: int = 1, **kwargs):
    return Sequential(mlp(*sizes, **kwargs), LazyLinear(osize))


class LazyResidBlock(LazyModuleMixin, Module):
    def __init__(self, embed_features: int = None, **kwargs):
        super().__init__()
        self._is_initted = False

        self.perc = Perceptron(embed_features, **kwargs)
        self.lin2 = LazyLinear(0, bias=True)

    def forward(self, val: Tensor):
        return val + self.lin2(self.perc(val))

    def initialize_parameters(self, val: Tensor):
        if not self._is_initted:
            if self.perc.lin.out_features is None:
                self.perc.lin.out_features = val.size(-1)
            self.lin2.out_features = val.size(-1)
            self._is_initted = True
