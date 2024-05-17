from math import sqrt
from typing import Iterable, Sequence

import attr
import torch
from torch import Tensor, LongTensor
from torch.nn import Module, ModuleList, Parameter, init, Sequential, Flatten, Unflatten, Linear

from . import lens_to_indptr_like
from .. import AttrsModule
from ..batched import BatchedConv1d


@attr.s
class TupleModule(AttrsModule):
    nets: Iterable[Module]
    lens: Sequence[int]
    dim: int = -2
    flatten: bool = True

    def __attrs_post_init__(self):
        self.nets = ModuleList(self.nets)

    def forward(self, x: Tensor) -> list[Tensor]:
        return [
            net(a.flatten(self.dim) if self.flatten else a)
            for net, a in zip(self.nets, x.split(self.lens, dim=self.dim))
        ]


class TupleLinear(Linear):
    lens: LongTensor

    def __init__(self, lens: LongTensor, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.register_buffer('lens', lens)
        self.in_features = self.lens.sum().item()
        self.out_features = out_features

        self.weight = Parameter(torch.empty((out_features, self.in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(len(self.lens), out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        from torch_scatter import segment_csr
        res = segment_csr(res := self.weight * x.unsqueeze(-2),
                          lens_to_indptr_like(res, self.lens, dim=-1)).movedim(-1, -2)
        return res if self.bias is None else res + self.bias


def channel_linear(in_channels, in_features, out_features, bias=True, device=None, dtype=None):
    return Sequential(
        BatchedConv1d(in_channels=in_channels, kernel_size=in_features,
                      out_channels=in_channels*out_features, groups=in_channels,
                      bias=bias, device=device, dtype=dtype),
        Flatten(-2), Unflatten(-1, (in_channels, out_features))
    )
