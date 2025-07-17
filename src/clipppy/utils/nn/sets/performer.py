from typing import Union, Callable, Mapping, TYPE_CHECKING

import attr
import torch
from performer_pytorch.performer_pytorch import Attention
from torch import Tensor
from torch.nn import Module, LayerNorm, Parameter, init

from phytorchx import broadcast_except
from phytorchx.attrs import AttrsModule, ParametrizedAttrsModule
from ..empty import _empty_module


class BatchedAttention(Attention):
    # noinspection PyMethodOverriding
    def forward(self, x, y):
        _x, _y = (
            _.flatten(end_dim=-3)
            for _ in broadcast_except(*(
                _.unsqueeze(0) if _.ndim==2 else _ for _ in (x, y)
            ), dim=-2)
        )
        return super().forward(x=_x, context=_y).reshape(
            torch.broadcast_shapes(x.shape[:-2], y.shape[:-2]) + x.shape[-2:]
        )


@attr.s(eq=False, auto_attribs=True)
class MAB(AttrsModule):
    embed_dim: int
    num_heads: int
    head_dim: int = None

    rFF: Union[Module, Callable[[Tensor], Tensor]] = _empty_module
    use_layer_norm: Union[bool, Mapping] = True

    def __attrs_post_init__(self):
        self.mha = BatchedAttention(self.embed_dim, heads=self.num_heads, dim_head=self.head_dim)
        self.ln_1, self.ln_2 = (
            LayerNorm(self.embed_dim, **({} if self.use_layer_norm is True else self.use_layer_norm))
            if self.use_layer_norm else _empty_module
            for _ in range(2))

    def forward(self, x: Tensor, y: Tensor):
        return self.ln_2((h := self.ln_1(x + self.mha(x, y))) + self.rFF(h))

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(eq=False, auto_attribs=True)
class SAB(AttrsModule):
    mab: MAB

    def forward(self, x: Tensor) -> Tensor:
        return self.mab(x, x)

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(eq=False, auto_attribs=True)
class PMA(ParametrizedAttrsModule):
    mab: MAB

    k: int = 1
    rFF: Union[Module, Callable[[Tensor], Tensor]] = _empty_module
    S: Parameter = attr.field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.S = Parameter(torch.empty((self.k, self.mab.embed_dim), **self.factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.S)

    def forward(self, z: Tensor) -> Tensor:
        return self.mab(self.S, self.rFF(z))

    if TYPE_CHECKING:
        __call__ = forward
