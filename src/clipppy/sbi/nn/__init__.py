from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Generic, Iterable, Mapping, TYPE_CHECKING, TypeVar, Union, Sequence

import attr
import torch
from more_itertools import always_iterable, consume, one, unique_everseen
from torch import nn, Tensor, LongTensor
from torch.nn import Module

from .._typing import _KT, _MultiKT, _SBIParamsT, _SBIObsT
from ..multi import dict_to_vect, PackerMixin
from ...utils.nn import LazyWhitenOnline
from ...utils.nn.attrs import AttrsModule
from ...utils.nn.empty import _empty_module


_HeadOoutT = TypeVar('_HeadOoutT')
_HeadOoutT2 = TypeVar('_HeadOoutT2')
_TailOutT = TypeVar('_TailOutT')


class BaseSBIHead(PackerMixin, AttrsModule, Generic[_HeadOoutT, _KT], ABC):
    @abstractmethod
    def forward(self, params: _SBIParamsT, obs: _SBIObsT) -> tuple[_SBIParamsT, _HeadOoutT]: ...

    if TYPE_CHECKING:
        __call__ = forward


@attr.s
class PassthroughSBIHead(BaseSBIHead[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    params_pre: Callable[[_SBIParamsT], _SBIParamsT] = attr.ib(default=_empty_module, kw_only=True)
    obs_pre: Callable[[_SBIObsT], _SBIObsT] = attr.ib(default=_empty_module, kw_only=True)

    def prepare_params(self, params: _SBIParamsT):
        return self.params_pre(params)

    def prepare_obs(self, obs: _SBIObsT):
        return self.obs_pre(obs)

    def forward(self, params, obs) -> tuple[_SBIParamsT, _SBIObsT]:
        return self.prepare_params(params), self.prepare_obs(obs)


@attr.s
class SBIHead(PassthroughSBIHead[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    head: Union[Module, Callable[[Tensor], _HeadOoutT]] = _empty_module

    whiten: bool = True

    def __attrs_post_init__(self):
        if self.whiten:
            self.head = nn.Sequential(LazyWhitenOnline(), self.head)

    def prepare_obs(self, obs: _SBIObsT) -> Tensor:
        return dict_to_vect(super().prepare_obs(obs), self.event_dims)

    def forward(self, params: _SBIParamsT, obs: _SBIObsT):
        return self.prepare_params(params), self.head(self.prepare_obs(obs))


@attr.s(auto_attribs=False)
class SetSBIHead(SBIHead[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    set_dim: int = 0

    if TYPE_CHECKING:
        head: Union[Module, Callable[[Tensor, Iterable[LongTensor]], _HeadOoutT]] = _empty_module

    def __attrs_post_init__(self):
        self.whitener = LazyWhitenOnline() if self.whiten else _empty_module

    def _nested_cat(self, nt: Sequence[Tensor]):
        return torch.cat(tuple(t.movedim(self.set_dim, 0) for t in nt), 0)
        # return torch.Tensor(nt.storage()).reshape(-1, *map(nt.size, range(2, nt.ndim)))

    def forward(self, params: _SBIParamsT, obs: Mapping[_KT, Sequence[Tensor]]):
        return self.params_pre(params), self.head(self.whitener(_obs := self.prepare_obs({
            key: self._nested_cat(val)
            for key, val in obs.items()
        })), (_obs.new_tensor(
            one(unique_everseen(tuple(_.shape[self.set_dim] for _ in v) for v in obs.values())),
            dtype=int
        ),))


@attr.s
class MultiSBIHead(BaseSBIHead[Mapping[Iterable[_KT], _HeadOoutT], _KT], Generic[_HeadOoutT, _KT]):
    heads: Mapping[Iterable[_KT], SBIHead[_HeadOoutT2, _KT]]
    post: Callable[[Mapping[Iterable[_KT], _HeadOoutT2]], _HeadOoutT]

    def __attrs_post_init__(self):
        for name, head in self.heads.items():
            self.register_module(str(name), head)
        self.event_dims = dict(item for head in self.heads.values() for item in head.event_dims.items())

    def forward(self, params: _SBIParamsT, obs: _SBIObsT) -> tuple[_SBIParamsT, _HeadOoutT]:
        return params, self.post({
            key: head(params, {k: obs[k] for k in key})[1]
            for key, head in self.heads.items()
        })


class BaseSBITail(AttrsModule, Generic[_HeadOoutT, _TailOutT, _KT], ABC):
    @abstractmethod
    def forward(self, params: _SBIParamsT, obs: _HeadOoutT, **kwargs) -> _TailOutT: ...

    if TYPE_CHECKING:
        __call__ = forward


class AbstractPackerSBITail(PackerMixin, BaseSBITail[_HeadOoutT, _TailOutT, _KT], Generic[_HeadOoutT, _TailOutT, _KT], ABC):
    pass


class ParamPackerSBITail(AbstractPackerSBITail[_HeadOoutT, _TailOutT, _KT], Generic[_HeadOoutT, _TailOutT, _KT]):
    @abstractmethod
    def _forward(self, theta: Tensor, x: _HeadOoutT, **kwargs) -> _TailOutT: ...

    def forward(self, params: _SBIParamsT, obs: _HeadOoutT, **kwargs) -> _TailOutT:
        return self._forward(self.pack(OrderedDict((k, params[k]) for k in self.param_names)), obs, **kwargs)

    if TYPE_CHECKING:
        __call__ = forward


@attr.s
class BaseMultiSBITail(BaseSBITail[_HeadOoutT, Mapping[_KT, _TailOutT], _KT], Generic[_HeadOoutT, _TailOutT, _KT]):
    tails: Mapping[_MultiKT, BaseSBITail[_HeadOoutT, _TailOutT, _KT]]

    def _add_tails(self, tails: Mapping[_MultiKT, Module], prefix=''):
        for key, mod in tails.items():
            setattr(self, prefix+(key if isinstance(key, str) else '_&_'.join(key)), mod)

    def __attrs_post_init__(self):
        self.tails = OrderedDict(self.tails)
        self._add_tails(self.tails)

        for key, tail in self.tails.items():
            if isinstance(tail, PackerMixin) and tail.param_names is None:
                tail.param_names = *always_iterable(key),

    def forward_one(self, key: _KT, params: _SBIParamsT, obs: _HeadOoutT, **kwargs) -> _TailOutT:
        return self.tails[key](params, obs, **kwargs)

    def forward(self, params: _SBIParamsT, obs: _HeadOoutT, **kwargs) -> Mapping[_KT, _TailOutT]:
        return {key: self.forward_one(key, params, obs, **kwargs) for key in self.tails.keys()}

    if TYPE_CHECKING:
        __call__ = forward


class MultiSBITail(BaseMultiSBITail[_HeadOoutT, _TailOutT, _KT], Generic[_HeadOoutT, _TailOutT, _KT]):
    pass
