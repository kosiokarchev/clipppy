from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Generic, Iterable, Mapping, TYPE_CHECKING, TypeVar, Union

import attr
import torch.nn
from more_itertools import always_iterable
from torch import nn, Tensor
from torch.nn import Module

from phytorchx.attrs import AttrsModule
from .._typing import _KT, _MultiKT, _SBIParamsT, _SBIObsT
from ..multi import PackerMixin
from ...utils.nn import LazyWhitenOnline
from ...utils.nn.empty import _empty_module

_HeadOoutT = TypeVar('_HeadOoutT')
_HeadOoutT2 = TypeVar('_HeadOoutT2')
_TailOutT = TypeVar('_TailOutT')


@attr.s(eq=False)
class ObsPacker(PackerMixin, AttrsModule):
    def forward(self, obs: _SBIObsT):
        return self.pack(OrderedDict((key, obs[key]) for key in self.obs_names))


class BaseSBIHead(PackerMixin, AttrsModule, Generic[_HeadOoutT, _KT], ABC):
    @abstractmethod
    def forward(self, params: _SBIParamsT, obs: _SBIObsT) -> tuple[_SBIParamsT, _HeadOoutT]: ...

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(eq=False, auto_attribs=True)
class PassthroughSBIHead(BaseSBIHead[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    obs_pre: Callable[[_SBIObsT], _SBIObsT] = attr.ib(default=_empty_module, kw_only=True)

    def prepare_obs(self, obs: _SBIObsT):
        return self.obs_pre(obs)

    def forward(self, params, obs) -> tuple[_SBIParamsT, _SBIObsT]:
        return params, self.prepare_obs(obs)


@attr.s(eq=False, auto_attribs=True)
class SBIHead(PassthroughSBIHead[_HeadOoutT, _KT], ObsPacker, Generic[_HeadOoutT, _KT]):
    head: Union[Module, Callable[[Tensor], _HeadOoutT]] = _empty_module

    whiten: bool = True

    def __attrs_post_init__(self):
        if self.whiten:
            self.head = nn.Sequential(LazyWhitenOnline(), self.head)

    def prepare_obs(self, obs: _SBIObsT) -> Tensor:
        return ObsPacker.forward(self, super().prepare_obs(obs))

    def forward(self, params: _SBIParamsT, obs: _SBIObsT):
        return params, self.head(self.prepare_obs(obs))


@attr.s(eq=False, auto_attribs=True)
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


@attr.s(eq=False, auto_attribs=True)
class ParamPackerSBITail(AbstractPackerSBITail[_HeadOoutT, _TailOutT, _KT], Generic[_HeadOoutT, _TailOutT, _KT]):
    params_pre: Callable[[_SBIParamsT], _SBIParamsT] = attr.ib(default=_empty_module, kw_only=True)

    @abstractmethod
    def _forward(self, theta: Tensor, x: _HeadOoutT, **kwargs) -> _TailOutT: ...

    def forward(self, params: _SBIParamsT, obs: _HeadOoutT, **kwargs) -> _TailOutT:
        params = self.params_pre({k: params[k] for k in self.param_names})
        return self._forward(self.pack(OrderedDict((k, params[k]) for k in self.param_names)), obs, **kwargs)

    if TYPE_CHECKING:
        __call__ = forward


class ModuleDict2(torch.nn.ModuleDict):
    sep = '_&_'

    def key_to_str(self, key: _MultiKT):
        return self.sep.join(always_iterable(key))

    def str_to_key(self, key: str):
        return tuple(key.split(self.sep)) if self.sep in key else key

    def __getitem__(self, item):
        return super().__getitem__(self.key_to_str(item))

    def __setitem__(self, key, value):
        return super().__setitem__(self.key_to_str(key), value)

    def __delitem__(self, key):
        return super().__delitem__(self.key_to_str(key))

    def __iter__(self):
        return self.keys()

    def __contains__(self, item):
        return super().__contains__(self.key_to_str(item))

    def keys(self):
        return (self.str_to_key(key) for key in super().keys())

    def items(self):
        return ((self.str_to_key(key), val) for key, val in super().items())


@attr.s(eq=False, auto_attribs=True)
class BaseMultiSBITail(BaseSBITail[_HeadOoutT, Mapping[_KT, _TailOutT], _KT], Generic[_HeadOoutT, _TailOutT, _KT]):
    tails: Mapping[_MultiKT, BaseSBITail[_HeadOoutT, _TailOutT, _KT]]

    # def _add_tails(self, tails: Mapping[_MultiKT, Module], prefix=''):
    #     for key, mod in tails.items():
    #         setattr(self, prefix+(key if isinstance(key, str) else '_&_'.join(key)), mod)

    def __attrs_post_init__(self):
        self.tails = ModuleDict2(self.tails)
        # self.tails = OrderedDict(self.tails)
        # self._add_tails(self.tails)

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
