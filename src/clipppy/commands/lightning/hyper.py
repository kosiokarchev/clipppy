from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Callable, Iterable, Mapping, Union

import numpy as np
from torch import Tensor
from torch.nn import Module

from .config import SchedulerConfig, schedulers as lrs
from ...utils.nn import linear, mlp, omlp
from ...utils.nn.empty import _empty_module

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
    tail: Union[Tail, BaseHParams] = field(default_factory=BaseHParams)


class ModuleHP(BaseHParams):
    def make(self, *args, **kwargs) -> Module:
        return _empty_module


@dataclass(repr=False)
class Tail(BaseHParams):
    thead: Union[MLP, BaseHParams] = field(default_factory=ModuleHP)
    xhead: BaseHParams = field(default_factory=ModuleHP)
    net: Union[OMLP, BaseHParams] = field(default_factory=ModuleHP)


@dataclass(repr=False)
class Linear(ModuleHP):
    size: int
    whiten: bool = True

    def make(self, **kwargs):
        return linear(self.size, whiten=self.whiten)


@dataclass(repr=False)
class MLP(ModuleHP):
    nlayers: int
    size: int
    osize: int = None
    whiten: bool = True

    def make(self, **kwargs):
        return mlp(
            *self.nlayers*(self.size,),
            *((self.osize,) if self.osize else ()),
            whiten=self.whiten)


@dataclass
class OMLP(MLP):
    osize: int = 1

    def make(self, **kwargs):
        return omlp(*self.nlayers*(self.size,), osize=self.osize, whiten=self.whiten)


class Scheduler(BaseHParams):
    cls: Callable[..., SchedulerConfig]

    def __init__(self, cls: Union[Callable[..., SchedulerConfig], str], **kwargs):
        super()._setattr__('cls', getattr(lrs, cls) if isinstance(cls, str) else cls)
        super().__init__(cls_=self.cls.__name__, **kwargs)

    def make(self, **kwargs):
        return self.cls(**dict(filter(lambda keyval: keyval[0] != 'cls_', self.items())), **kwargs)


@dataclass(repr=False)
class Training(BaseHParams):
    lr: float
    batch_size: int
    scheduler: Scheduler = None
