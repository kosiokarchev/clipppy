from __future__ import annotations

from typing import Literal, TYPE_CHECKING, Union

import torch
import attr
from torch import nn, Tensor
from torch.nn import Module

from phytorch.utils.broadcast import broadcast_cat

from ...sbi.nn import BaseSBITail
from ...utils.nn import LazyWhitenOnline
from ...utils.nn.empty import _empty_module

__all__ = (
    'NRETail', 'WhiteningTail', 'UWhiteningTail', 'IUWhiteningTail'
)


@attr.s(auto_attribs=True, eq=False)
class NRETail(BaseSBITail[Tensor, Tensor, Tensor]):
    net: Module = attr.ib(default=_empty_module)
    thead: Module = attr.ib(default=_empty_module)
    xhead: Module = attr.ib(default=_empty_module)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        ts = (self.thead(theta), self.xhead(x))
        shape = torch.broadcast_shapes(*(t.shape[:-1] for t in ts))
        return self.net(torch.cat(tuple(t.expand(*shape, t.shape[-1]) for t in ts), -1)).squeeze(-1)

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class WhiteningTail(NRETail):
    def __attrs_post_init__(self):
        self.thead = nn.Sequential(LazyWhitenOnline(), self.thead)


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
