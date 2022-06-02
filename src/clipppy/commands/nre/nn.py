from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Generic, Iterable, Literal, Mapping, TYPE_CHECKING, TypeVar, Union

import torch
import attr
from more_itertools import always_iterable, consume
from torch import nn, Size, Tensor
from torch.nn import Module

from phytorch.utils.broadcast import broadcast_cat

from ...utils.nn import _empty_module, WhitenOnline


__all__ = (
    'BaseNREHead', 'NREHead', 'WhiteningHead',
    'BaseNRETail', 'NRETail', 'WhiteningTail', 'MultiNRETail',
    'UWhiteningTail', 'IUWhiteningTail'
)


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


class BaseNREHead(AttrsModule, Generic[_HeadPoutT, _HeadOoutT, _KT], ABC):
    @abstractmethod
    def forward(self, params: Mapping[_KT, Tensor], obs: Mapping[_KT, Tensor]) -> tuple[_HeadPoutT, _HeadOoutT]: ...

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class NREHead(BaseNREHead[Mapping[_KT, Tensor], _HeadOoutT, _KT]):
    head: Union[Module, Callable[[Tensor], _HeadOoutT]] = _empty_module
    event_dims: Mapping[_KT, int] = attr.ib(factory=dict)

    def forward(self, params: Mapping[_KT, Tensor], obs: Mapping[_KT, Tensor]):
        return params, self.head(dict_to_vect(obs, self.event_dims))


@attr.s(auto_attribs=True, eq=False)
class WhiteningHead(NREHead):
    def __attrs_post_init__(self):
        self.head = nn.Sequential(WhitenOnline(), self.head)


class BaseNRETail(AttrsModule, Generic[_HeadPoutT, _HeadOoutT, _TailOutT, _KT], ABC):
    @abstractmethod
    def forward(self, theta: _HeadPoutT, x: _HeadOoutT) -> _TailOutT: ...

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class NRETail(BaseNRETail[Tensor, Tensor, Tensor, _KT]):
    net: Module = attr.ib(default=_empty_module)
    thead: Module = attr.ib(default=_empty_module)
    xhead: Module = attr.ib(default=_empty_module)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        ts = (self.thead(theta), self.xhead(x))
        shape = torch.broadcast_shapes(*(t.shape[:-1] for t in ts))
        return self.net(torch.cat(tuple(t.expand(*shape, t.shape[-1]) for t in ts), -1)).squeeze(-1)


@attr.s(auto_attribs=True, eq=False)
class WhiteningTail(NRETail):
    def __attrs_post_init__(self):
        self.thead = nn.Sequential(WhitenOnline(), self.thead)


class UWhiteningTail(WhiteningTail):
    def forward(self, theta: Tensor, x: tuple[Tensor, Tensor]):
        return super().forward(theta, x[1])


@attr.s(auto_attribs=True, eq=False)
class IUWhiteningTail(UWhiteningTail):
    ihead: Module = attr.ib(default=_empty_module)
    shead: Module = attr.ib(default=_empty_module)

    additional: Union[Tensor, Literal[False]] = None
    subsample: int = None
    summarize: bool = False

    def forward(self, theta: Tensor, x: tuple[Tensor, Tensor]) -> Tensor:
        args = self.thead(theta), self.xhead(x[0])
        if self.additional is not False:
            args += self.ihead(
                self.additional if torch.is_tensor(self.additional)
                else torch.linspace(-1, 1, theta.shape[-1]).unsqueeze(-1),
            ),
        if self.summarize:
            args += self.shead(x[1].unsqueeze(-2)),

        y = broadcast_cat(args, -1)

        if self.training and self.subsample is not None:
            y = y[..., torch.randint(y.shape[-2], (self.subsample,)), :]

        return self.net(y).squeeze(-1)


@attr.s(auto_attribs=True, eq=False)
class MultiNRETail(BaseNRETail[Mapping[_KT, Tensor], _HeadOoutT, Iterable[Tensor], _KT], ParamPackerMixin):
    tails: Mapping[Union[_KT, Iterable[_KT]], BaseNRETail[Tensor, _HeadOoutT]] = attr.ib(factory=dict)

    def __attrs_post_init__(self):
        if not isinstance(self.tails, Module):
            self.tails = OrderedDict(self.tails)
            consume(setattr(self, key if isinstance(key, str) else '_&_'.join(key), val) for key, val in self.tails.items())

    def forward_one(self, key: _KT, params: Mapping[_KT, Tensor], x: _HeadOoutT) -> Tensor:
        return self.tails[key](self.pack(OrderedDict((k, params[k]) for k in always_iterable(key))), x)

    def forward(self, params: Mapping[_KT, Tensor], x: _HeadOoutT) -> Iterable[Tensor]:
        return {key: self.forward_one(key, params, x) for key in self.tails.keys()}
