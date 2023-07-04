from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Generic, Iterable, Mapping, TYPE_CHECKING, TypeVar, Union, Sequence

import attr
import torch
from more_itertools import always_iterable, consume, one, unique_everseen
from torch import nn, Tensor, LongTensor
from torch.nn import Module

from ._typing import _KT, _MultiKT
from .multi import dict_to_vect, ParamPackerMixin
from ..utils.nn import LazyWhitenOnline
from ..utils.nn.attrs import AttrsModule
from ..utils.nn.empty import _empty_module


_HeadPoutT = TypeVar('_HeadPoutT')
_HeadOoutT = TypeVar('_HeadOoutT')
_HeadOoutT2 = TypeVar('_HeadOoutT2')
_TailOutT = TypeVar('_TailOutT')


class BaseSBIHead(AttrsModule, Generic[_HeadPoutT, _HeadOoutT, _KT], ABC):
    # @abstractmethod
    event_dims: Mapping[_KT, int]

    @abstractmethod
    def forward(self, params: Mapping[_KT, Tensor], obs: Mapping[_KT, Tensor]) -> tuple[_HeadPoutT, _HeadOoutT]: ...

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class SBIHead(BaseSBIHead[Mapping[_KT, Tensor], _HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    head: Union[Module, Callable[[Tensor], _HeadOoutT]] = _empty_module
    event_dims: Mapping[_KT, int] = attr.ib(factory=dict)

    whiten: bool = True

    def __attrs_post_init__(self):
        if self.whiten:
            self.head = nn.Sequential(LazyWhitenOnline(), self.head)

    def prepare_obs(self, obs: Mapping[_KT, Tensor]) -> Tensor:
        return dict_to_vect(obs, self.event_dims)

    def forward(self, params: Mapping[_KT, Tensor], obs: Mapping[_KT, Tensor]):
        return params, self.head(self.prepare_obs(obs))


@attr.s(eq=False)
class SetSBIHead(SBIHead[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    set_dim: int = 0

    if TYPE_CHECKING:
        head: Union[Module, Callable[[Tensor, Iterable[LongTensor]], _HeadOoutT]] = _empty_module

    def __attrs_post_init__(self):
        self.whitener = LazyWhitenOnline() if self.whiten else _empty_module

    def _nested_cat(self, nt: Sequence[Tensor]):
        return torch.cat(tuple(t.movedim(self.set_dim, 0) for t in nt), 0)
        # return torch.Tensor(nt.storage()).reshape(-1, *map(nt.size, range(2, nt.ndim)))

    def forward(self, params: Mapping[_KT, Tensor], obs: Mapping[_KT, Sequence[Tensor]]):
        return params, self.head(self.whitener(_obs := self.prepare_obs({
            key: self._nested_cat(val)
            for key, val in obs.items()
        })), (_obs.new_tensor(
            one(unique_everseen(tuple(_.shape[self.set_dim] for _ in v) for v in obs.values())),
            dtype=int
        ),))


# TODO: deprecate and remove WhiteningHead
@attr.s(auto_attribs=True, eq=False)
class WhiteningHead(SBIHead):
    pass


@attr.s(auto_attribs=True, eq=False)
class MultiSBIHead(BaseSBIHead[Mapping[_KT, Tensor], Mapping[Iterable[_KT], _HeadOoutT], _KT], Generic[_HeadOoutT, _KT]):
    heads: Mapping[Iterable[_KT], SBIHead[_HeadOoutT2, _KT]]
    post: Callable[[Mapping[Iterable[_KT], _HeadOoutT2]], _HeadOoutT]

    def __attrs_post_init__(self):
        for name, head in self.heads.items():
            self.register_module(str(name), head)
        self.event_dims = dict(item for head in self.heads.values() for item in head.event_dims.items())

    def forward(self, params: Mapping[_KT, Tensor], obs: Mapping[_KT, Tensor]) -> tuple[Mapping[
        _KT, Tensor], _HeadOoutT]:
        return params, self.post({
            key: head(params, {k: obs[k] for k in key})[1]
            for key, head in self.heads.items()
        })


class BaseSBITail(AttrsModule, Generic[_HeadPoutT, _HeadOoutT, _TailOutT], ABC):
    @abstractmethod
    def forward(self, theta: _HeadPoutT, x: _HeadOoutT, **kwargs) -> _TailOutT: ...

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class MultiSBITail(BaseSBITail[Mapping[_KT, Tensor], _HeadOoutT, Mapping[_KT, _TailOutT]], ParamPackerMixin, Generic[_HeadOoutT, _TailOutT, _KT]):
    tails: Mapping[_MultiKT, BaseSBITail[Tensor, _HeadOoutT, _TailOutT, _KT]] = attr.ib(factory=dict)

    def __attrs_post_init__(self):
        if not isinstance(self.tails, Module):
            self.tails = OrderedDict(self.tails)
            consume(setattr(self, key if isinstance(key, str) else '_&_'.join(key), val) for key, val in self.tails.items())

    def forward_one(self, key: _KT, params: Mapping[_KT, Tensor], x: _HeadOoutT, **kwargs) -> _TailOutT:
        return self.tails[key](self.pack(OrderedDict((k, params[k]) for k in always_iterable(key))), x, **kwargs)

    def forward(self, params: Mapping[_KT, Tensor], x: _HeadOoutT, **kwargs) -> Mapping[_KT, _TailOutT]:
        return {key: self.forward_one(key, params, x, **kwargs) for key in self.tails.keys()}

    if TYPE_CHECKING:
        __call__ = forward


class BaseGASBITail(BaseSBITail[_HeadPoutT, _HeadOoutT, _TailOutT], ABC):
    @abstractmethod
    def sim_log_prob_grad(self, theta: _HeadPoutT): ...


@attr.s(auto_attribs=True, eq=False)
class BaseMultiGASBITail(BaseGASBITail[Mapping[_KT, Tensor], _HeadOoutT, Mapping[_KT, _TailOutT]],
                         ParamPackerMixin, Generic[_HeadOoutT, _TailOutT, _KT]):
    def sim_log_prob_grad_one(self, key: _KT, params: Mapping[_KT, Tensor]):
        return self.pack(OrderedDict((k, params[k].grad) for k in always_iterable(key)))

    def sim_log_prob_grad(self, params: Mapping[_KT, Tensor]):
        return {key: self.sim_log_prob_grad_one(key, params) for key in self.tails.keys()}

    def forward(self, theta: _HeadPoutT, x: _HeadOoutT, **kwargs) -> _TailOutT:
        return super().forward(theta, x, **kwargs)


class MultiGASBITail(MultiSBITail[_HeadOoutT, _TailOutT, _KT],
                     BaseMultiGASBITail[_HeadOoutT, _TailOutT, _KT],
                     Generic[_HeadOoutT, _TailOutT, _KT]):
    pass
