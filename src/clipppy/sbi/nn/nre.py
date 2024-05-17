from __future__ import annotations

from abc import ABC
from collections import OrderedDict
from itertools import chain
from typing import Literal, Union, Generic, TYPE_CHECKING

import attr
import torch
from torch import nn, Tensor
from torch.nn import Module

from phytorchx import broadcast_cat
from . import BaseSBITail, _HeadOoutT, ParamPackerSBITail, AbstractPackerSBITail
from .._typing import _KT, _SBIObsT, _SBIParamsT
from ...utils.nn.empty import _empty_module


class BaseNRETail(BaseSBITail[_HeadOoutT, Tensor, _KT], Generic[_HeadOoutT, _KT], ABC):
    pass


class AbstractPackerNRETail(AbstractPackerSBITail[_HeadOoutT, Tensor, _KT], BaseNRETail[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT], ABC):
    pass


class ParamPackerNRETail(ParamPackerSBITail[_HeadOoutT, Tensor, _KT], AbstractPackerNRETail[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT], ABC):
    pass


@attr.s
class SimpleNRETail(AbstractPackerNRETail[_SBIObsT, _KT], Generic[_KT]):
    net: Module = attr.ib(default=_empty_module)

    def forward(self, params: _SBIParamsT, obs: _SBIObsT, **kwargs) -> Tensor:
        return self.net(self.pack(OrderedDict(chain(
            ((k, params[k]) for k in self.param_names),
            ((k, obs[k]) for k in self.obs_names)
        ))), **kwargs).squeeze(-1)

    if TYPE_CHECKING:
        __call__ = forward


@attr.s
class NRETail(ParamPackerNRETail[Tensor, _KT], SimpleNRETail[_KT], Generic[_KT]):
    thead: Module = attr.ib(default=_empty_module)
    xhead: Module = attr.ib(default=_empty_module)

    def _forward(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        return self.net(broadcast_cat((self.thead(theta), self.xhead(x)), -1)).squeeze(-1)


class UNRETail(NRETail[_KT], Generic[_KT]):
    def _forward(self, theta: Tensor, x: tuple[Tensor, Tensor], **kwargs):
        return super()._forward(theta, x[1], **kwargs)


@attr.s
class IUNRETail(UNRETail[_KT], Generic[_KT]):
    ihead: Module = attr.ib(default=_empty_module)
    shead: Module = attr.ib(default=_empty_module)

    _additional: Union[Tensor, Literal[False]] = attr.ib(default=None, repr=False)
    subsample: int = None
    summarize: bool = False

    def get_additional(self, hint):
        if not hasattr(self, 'additional'):
            self.register_buffer(
                'additional',
                self._additional if torch.is_tensor(self._additional)
                else torch.linspace(-1, 1, hint.shape[-2], device=hint.device, dtype=hint.dtype).unsqueeze(-1)
            )
        return self.additional

    def _forward(self, theta: Tensor, x: tuple[Tensor, Tensor], **kwargs) -> Tensor:
        args = self.thead(theta), self.xhead(x[0])
        if self._additional is not False:
            args += self.ihead(self.get_additional(args[0])),
        if self.summarize:
            args += self.shead(x[1].unsqueeze(-2)),

        y = broadcast_cat(args, -1)

        if self.training and self.subsample is not None:
            y = y[..., torch.randint(y.shape[-2], (self.subsample,)), :]

        return self.net(y).squeeze(-1)
