from __future__ import annotations

from typing import Union, Callable, TYPE_CHECKING

import attr
import torch
from torch import Tensor
from torch.nn import Module, Parameter, init

from ..attrs import AttrsModule, ParametrizedAttrsModel
from ..batched import BatchedMultiheadAttention
from ..empty import _empty_module


@attr.s(auto_attribs=True, eq=False)
class MAB(AttrsModule):
    embed_dim: int
    num_heads: int

    net: Union[Module, Callable[[Tensor], Tensor]] = _empty_module

    def __attrs_post_init__(self):
        self.mha = BatchedMultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

    def forward(self, x, y) -> Tensor:
        # return LayerNorm((h := LayerNorm(x + self.mha(x, y, y))) + self.net(h))
        return (h := x + self.mha(x, y, y, need_weights=False)[0]) + self.net(h)

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class SAB(AttrsModule):
    mab: MAB

    def forward(self, x: Tensor) -> Tensor:
        return self.mab(x, x)

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class ISAB(ParametrizedAttrsModel):
    m: int
    mab_1: MAB
    mab_2: MAB

    def __attrs_post_init__(self):
        assert self.mab_1.embed_dim == self.mab_2.embed_dim
        self.I = Parameter(torch.empty((self.m, self.mab_1.embed_dim), **self.factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.I)

    def forward(self, x: Tensor) -> Tensor:
        return self.mab_2(x, self.mab_1(self.I, x))

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class PMA(ParametrizedAttrsModel):
    k: int
    mab: MAB
    net: Union[Module, Callable[[Tensor], Tensor]] = _empty_module

    def __attrs_post_init__(self):
        self.S = Parameter(torch.empty((self.k, self.mab.embed_dim), **self.factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.S)

    def forward(self, z):
        return self.mab(self.S, self.net(z))

    if TYPE_CHECKING:
        __call__ = forward
