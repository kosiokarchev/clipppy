from __future__ import annotations

from functools import partial
from itertools import cycle
from typing import Iterable, Literal, Protocol, Union, TYPE_CHECKING

import attr
import torch
from more_itertools import last, always_iterable
from torch import Tensor, LongTensor
from torch.nn import ModuleList, Module

from phytorchx import broadcast_gather, broadcast_cat

from ..attrs import AttrsModule


def lens_to_indptr_like(t: Tensor, lens: LongTensor, dim: int) -> LongTensor:
    return broadcast_cat((lens.new_zeros((1,)), lens.cumsum(-1)), -1).to(t.device).expand(*t.shape[:dim % t.ndim], -1)


class _collapse_fn_t(Protocol):
    def __call__(self, t: Tensor, indptr: Iterable[int]) -> Tensor: ...


def _collapse(t: Tensor, indptr: Tensor, reduce: Literal['mean', 'sum']):
    from torch_scatter import segment_csr
    return segment_csr(t, indptr, reduce=reduce)


collapse_sum = partial(_collapse, reduce='sum')
collapse_mean = partial(_collapse, reduce='mean')


@attr.s(auto_attribs=True, eq=False)
class NestedSetsProcessor(AttrsModule):
    nets: Iterable[Module]
    collapse_fns: Iterable[_collapse_fn_t] = (collapse_mean,)
    append_lens: bool = True

    def __attrs_post_init__(self):
        if not isinstance(self.nets, ModuleList):
            self.nets = ModuleList(self.nets)

    def iter_levels(self, lenss: Iterable[LongTensor]) -> tuple[Iterable[LongTensor], Module, _collapse_fn_t]:
        yield from zip(lenss, self.nets, cycle(self.collapse_fns))

    @staticmethod
    def forward_one(t: Tensor, lens: LongTensor, net: Module, collapse_fn: _collapse_fn_t, *args, dim: int) -> Tensor:
        return collapse_fn(net(t), lens_to_indptr_like(t, lens, dim))

    def _append_lens(self, t: Tensor, lens: LongTensor, dim: int) -> Tensor:
        if self.append_lens:
            assert dim % t.ndim - t.ndim == -2
            return broadcast_cat((t, lens.to(t).unsqueeze(-1)), -1)
        else:
            return t

    def forward(self, t: Tensor, lenss: Iterable[LongTensor], dim=-2) -> Tensor:
        return last(
            t for t in [t]
            for lens, *args in self.iter_levels(lenss)
            for t in [self._append_lens(self.forward_one(t, lens, *args, dim=dim), lens, dim)]
        )

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class SubsampledNestedSetsProcessor(NestedSetsProcessor):
    subsample_counts: Union[Union[int, None], Iterable[Union[int, None]]] = None
    subsample_fracs: Union[float, Iterable[float]] = (1,)

    def iter_levels(self, lenss: Iterable[LongTensor]) -> tuple[Iterable[LongTensor], Module, _collapse_fn_t, int]:
        for (lens, net, collapse_fn), count, frac in zip(
            super().iter_levels(lenss),
            cycle(always_iterable(self.subsample_counts) if self.subsample_counts is not None else (None,)),
            cycle(always_iterable(self.subsample_fracs))
        ):
            yield lens, net, collapse_fn, (count if count is not None else int((frac * lens.sum(-1)).ceil()))

    @staticmethod
    def get_idx(t: Tensor, subc: int, dim: int) -> Tensor:
        return torch.randint(t.shape[dim], (*t.shape[:dim], subc), device=t.device)

    def subsample(self, t: Tensor, indptr: Iterable[int], subc: int, dim: int) -> tuple[Tensor, LongTensor]:
        return (
            broadcast_gather(t, dim, idx := self.get_idx(t, subc, dim)),
            torch.logical_and(idx.unsqueeze(-1) >= indptr[..., None, :-1],
                              idx.unsqueeze(-1) < indptr[..., None, 1:]).sum(-2)
        )

    def forward_one(self, t: Tensor, lens: LongTensor, net: Module, collapse_fn: _collapse_fn_t, subc: int = None, *args, dim: int):
        if self.training:
            newt, newlens = self.subsample(t, lens_to_indptr_like(t, lens, dim), subc, dim)
            res = super().forward_one(newt, newlens, net, collapse_fn, dim=dim)
            if collapse_fn is collapse_sum:
                res = res * (lens / newlens).unsqueeze(-1).unflatten(-1, (t.ndim - dim % t.ndim)*(1,))
            return res
        else:
            return super().forward_one(t, lens, net, collapse_fn, dim=dim)
