from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Callable, Iterable, Mapping, Any, Generic, TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from .config import SchedulerConfig, schedulers as lrs, OptimizerConfig
from ...utils.nn import mlp, omlp, Perceptron as nn_Perceptron
from ...utils.nn.empty import _empty_module
from ...utils.nn.sets.transformer import MAB as nn_MAB, SAB as nn_SAB, ISAB as nn_ISAB, PMA as nn_PMA


lrmap = {'highest': 1e-2, 'high': 1e-3, 'low': 1e-4, 'lowest': 1e-5, 'snail': 1e-6}


_T = TypeVar('_T')
_non_iterables = (str, np.ndarray, Tensor)


def nested_iterables(o, keys=()):
    yield from (chain(*(
        nested_iterables(v, keys + (k,))
        for k, v in (o.items() if isinstance(o, Mapping) else enumerate(o))
    )) if isinstance(o, Iterable) and not isinstance(o, _non_iterables) else ((keys, o),))


class FlatAttrDict(dict):
    _setattr__ = dict.__setattr__
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        return super().__getattribute__(item)

    def collapse(self, delimiter='/'):
        for key, val in self.items():
            if isinstance(val, FlatAttrDict):
                for k, v in val.collapse(delimiter=delimiter):
                    d = delimiter.join((str(key), str(k)))
                    yield d, v
            else:
                yield key, val


class BaseHParams(FlatAttrDict):
    def make(self, **kwargs):
        raise NotImplementedError


@dataclass(repr=False)
class Hyperparams(BaseHParams):
    structure: Structure
    training: Training


@dataclass(repr=False)
class Structure(BaseHParams):
    head: BaseHParams = field(default_factory=BaseHParams)
    tail: Tail | BaseHParams = field(default_factory=BaseHParams)


class ModuleHP(BaseHParams):
    def make(self, *args, **kwargs) -> Module:
        return _empty_module


@dataclass(repr=False)
class Tail(BaseHParams):
    thead: MLP | BaseHParams = field(default_factory=ModuleHP)
    xhead: BaseHParams = field(default_factory=ModuleHP)
    net: OMLP | BaseHParams = field(default_factory=ModuleHP)


@dataclass(repr=False)
class Perceptron(ModuleHP):
    size: int
    kwargs: dict = field(default_factory=dict)

    def make(self, **kwargs):
        return nn_Perceptron(self.size, **self.kwargs)


@dataclass(repr=False)
class MLP(ModuleHP):
    nlayers: int
    size: int
    osize: int = None

    kwargs: dict = field(default_factory=dict)

    def make(self, **kwargs):
        return mlp(
            *self.nlayers*(self.size,),
            *((self.osize,) if self.osize else ()),
            **self.kwargs
        )


@dataclass
class OMLP(MLP):
    osize: int = 1

    def make(self, **kwargs):
        return omlp(*self.nlayers*(self.size,), osize=self.osize, **self.kwargs)


@dataclass
class MAB(ModuleHP):
    embed_dim: int
    num_heads: int = 1
    rFF: ModuleHP = field(default_factory=ModuleHP)
    use_layer_norm: bool = True

    def make(self, **kwargs) -> nn_MAB:
        return nn_MAB(self.embed_dim, self.num_heads, self.rFF.make(), self.use_layer_norm)


@dataclass
class SAB(ModuleHP):
    mab: MAB

    def make(self, **kwargs) -> nn_SAB:
        return nn_SAB(self.mab.make(**kwargs))


@dataclass
class ISAB(ModuleHP):
    m: int
    mab_1: MAB
    mab_2: MAB

    def make(self, **kwargs) -> nn_ISAB:
        return nn_ISAB(self.m, self.mab_1.make(**kwargs), self.mab_2.make(**kwargs), **kwargs)


@dataclass
class PMA(ModuleHP):
    mab: MAB
    k: int = 1
    rFF: ModuleHP = field(default_factory=ModuleHP)

    def make(self, **kwargs) -> nn_PMA:
        return nn_PMA(self.mab.make(**kwargs), self.k, self.rFF.make(**kwargs), **kwargs)


class ObjectParams(BaseHParams, Generic[_T]):
    cls: Callable[..., _T]
    _namespace: Any

    def __init__(self, cls: Callable[..., _T] | str, **kwargs):
        super()._setattr__('cls', getattr(self._namespace, cls) if isinstance(cls, str) else cls)
        super().__init__(cls_=self.cls.__name__, **kwargs)

    def make(self, **kwargs):
        return self.cls(**dict(filter(lambda keyval: keyval[0] != 'cls_', self.items())), **kwargs)


class Optimizer(ObjectParams[OptimizerConfig]):
    _namespace = torch.optim

    def make(self, **kwargs):
        return OptimizerConfig(self.cls, kwargs=dict(filter(lambda keyval: keyval[0] != 'cls_', self.items())) | kwargs)


class Scheduler(ObjectParams[SchedulerConfig]):
    _namespace = lrs


@dataclass(repr=False)
class Training(BaseHParams):
    lr: float
    batch_size: int
    optimizer: Optimizer = None
    scheduler: Scheduler = None
