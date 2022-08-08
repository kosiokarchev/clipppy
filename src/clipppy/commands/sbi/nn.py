from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Generic, Iterable, Mapping, TYPE_CHECKING, TypeVar, Union

import attr
import torch
from more_itertools import always_iterable, consume
from torch import nn, Size, Tensor
from torch.nn import Module

from clipppy.utils.nn import _empty_module, WhitenOnline


_KT = TypeVar('_KT')
_HeadPoutT = TypeVar('_HeadPoutT')
_HeadOoutT = TypeVar('_HeadOoutT')
_TailOutT = TypeVar('_TailOutT')


def dict_to_vect(d: Mapping[_KT, Tensor], ndims: Mapping[_KT, int]) -> Tensor:
    return torch.cat(tuple(
        v.flatten(-ndim) if ndim else v.unsqueeze(-1)
        for k, v in d.items() for ndim in [ndims.get(k, 0)]
    ), -1)


def vect_to_dict(v: Tensor, shapes: Mapping[_KT, Size]) -> Mapping[_KT, Tensor]:
    return OrderedDict(
        (key, val) for i in [0] for key, shape in shapes.items()
        for j in [i+shape.numel()]
        for val in [v[..., i:j].reshape(*v.shape[:-1], *shape)]
        for i in [j]
    )


@attr.s(auto_attribs=True, eq=False)
class ParamPackerMixin:
    param_event_dims: Mapping[_KT, int] = attr.ib(factory=dict)

    def pack(self, d: Mapping[_KT, Tensor]) -> Tensor:
        return dict_to_vect(d, self.param_event_dims)

    def unpack(self, d: Mapping[_KT, Tensor], v: Tensor):
        return vect_to_dict(v, OrderedDict(
            (key, val.shape[val.ndim-self.param_event_dims.get(key, 0):])
            for key, val in d.items()
        ))


@attr.s(eq=False)
class AttrsModule(Module):
    def __attrs_pre_init__(self):
        super().__init__()


class BaseSBIHead(AttrsModule, Generic[_HeadPoutT, _HeadOoutT, _KT], ABC):
    @abstractmethod
    def forward(self, params: Mapping[_KT, Tensor], obs: Mapping[_KT, Tensor]) -> tuple[_HeadPoutT, _HeadOoutT]: ...

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class SBIHead(BaseSBIHead[Mapping[_KT, Tensor], _HeadOoutT, _KT]):
    head: Union[Module, Callable[[Tensor], _HeadOoutT]] = _empty_module
    event_dims: Mapping[_KT, int] = attr.ib(factory=dict)

    def prepare_obs(self, obs: Mapping[_KT, Tensor]) -> Tensor:
        return dict_to_vect(obs, self.event_dims)

    def forward(self, params: Mapping[_KT, Tensor], obs: Mapping[_KT, Tensor]):
        return params, self.head(self.prepare_obs(obs))


@attr.s(auto_attribs=True, eq=False)
class WhiteningHead(SBIHead):
    def __attrs_post_init__(self):
        self.head = nn.Sequential(WhitenOnline(), self.head)


class BaseSBITail(AttrsModule, Generic[_HeadPoutT, _HeadOoutT, _TailOutT], ABC):
    @abstractmethod
    def forward(self, theta: _HeadPoutT, x: _HeadOoutT) -> _TailOutT: ...

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class MultiSBITail(BaseSBITail[Mapping[_KT, Tensor], _HeadOoutT, Mapping[_KT, _TailOutT]], ParamPackerMixin, Generic[_HeadOoutT, _TailOutT, _KT]):
    tails: Mapping[Union[_KT, Iterable[_KT]], BaseSBITail[Tensor, _HeadOoutT, _TailOutT, _KT]] = attr.ib(factory=dict)

    def __attrs_post_init__(self):
        if not isinstance(self.tails, Module):
            self.tails = OrderedDict(self.tails)
            consume(setattr(self, key if isinstance(key, str) else '_&_'.join(key), val) for key, val in self.tails.items())

    def forward_one(self, key: _KT, params: Mapping[_KT, Tensor], x: _HeadOoutT) -> _TailOutT:
        return self.tails[key](self.pack(OrderedDict((k, params[k]) for k in always_iterable(key))), x)

    def forward(self, params: Mapping[_KT, Tensor], x: _HeadOoutT) -> Mapping[_KT, _TailOutT]:
        return {key: self.forward_one(key, params, x) for key in self.tails.keys()}

    if TYPE_CHECKING:
        __call__ = forward
