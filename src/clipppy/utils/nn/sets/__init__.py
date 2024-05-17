from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from itertools import cycle
from typing import Iterable, Literal, Protocol, Union, TYPE_CHECKING, Callable

import attr
import torch
from more_itertools import last, always_iterable
from torch import Tensor, LongTensor
from torch.nn import ModuleList, Module

from phytorchx import broadcast_gather, broadcast_cat
from ..attrs import AttrsModule


class BatchedSetModule(Module, ABC):
    @abstractmethod
    def forward(self, t: Tensor, lens: LongTensor, dim: int = -2) -> Tensor: ...


@attr.s
class ForSet(BatchedSetModule, AttrsModule):
    net: Union[Module, Callable[[Tensor], Tensor]]

    def forward(self, t: Tensor, lens: LongTensor, dim: int = -2) -> Tensor:
        return torch.stack(tuple(
            self.net(st) for st in t.split(lens.tolist())
        ), dim=dim)


@attr.s
class ForDataParallel(AttrsModule):
    mod: Module
    batch_size: int
    batch_dim: int = 0

    def forward(self, t: Tensor):
        return torch.cat([
            self.mod(_t) for _t in t.split(self.batch_size, dim=self.batch_dim)
        ], self.batch_dim)


def lens_to_indptr_like(t: Tensor, lens: LongTensor, dim: int) -> LongTensor:
    return broadcast_cat((lens.new_zeros((1,)), lens.cumsum(-1)), -1).to(t.device).expand(*t.shape[:dim % t.ndim], -1)


class _collapse_fn_t(Protocol):
    def __call__(self, t: Tensor, indptr: Iterable[int]) -> Tensor: ...


def _collapse(t: Tensor, indptr: Tensor, reduce: Literal['mean', 'sum']):
    from torch_scatter import segment_csr
    return segment_csr(t, indptr, reduce=reduce)


collapse_sum = partial(_collapse, reduce='sum')
collapse_mean = partial(_collapse, reduce='mean')


@attr.s(kw_only=True)
class Collapser(AttrsModule):
    append_lens: bool = True

    def _append_lens(self, t: Tensor, lens: LongTensor, dim: int) -> Tensor:
        if self.append_lens:
            assert dim % t.ndim - t.ndim == -2
            return broadcast_cat((t, lens.to(t).unsqueeze(-1)), -1)
        else:
            return t


@attr.s
class SetCollapser(Collapser, BatchedSetModule):
    net: Union[Module, Callable[[Tensor], Tensor]]
    collapse_fn: _collapse_fn_t = collapse_mean

    def forward(self, t: Tensor, lens: LongTensor, dim=-2):
        return self._append_lens(self.collapse_fn(self.net(t), lens_to_indptr_like(t, lens, dim)), lens, dim)


@attr.s
class SubsampledSetCollapser(SetCollapser):
    subsample_frac: float = attr.ib(kw_only=True)

    @staticmethod
    def get_idx(t: Tensor, subc: int, dim: int) -> Tensor:
        return torch.randint(t.shape[dim], (*t.shape[:dim], subc), device=t.device)

    def subsample(self, t: Tensor, indptr: Iterable[int], subc: int, dim: int) -> tuple[Tensor, LongTensor]:
        return (
            broadcast_gather(t, dim, idx := self.get_idx(t, subc, dim)),
            torch.logical_and(idx.unsqueeze(-1) >= indptr[..., None, :-1],
                              idx.unsqueeze(-1) < indptr[..., None, 1:]).sum(-2)
        )

    def forward(self, t: Tensor, lens: LongTensor, dim=-2):
        if self.training:
            subt, sublens = self.subsample(t, lens_to_indptr_like(t, lens, dim), int((self.subsample_frac * lens.sum(-1)).ceil()), dim)
            res = self.collapse_fn(self.net(subt), lens_to_indptr_like(subt, sublens, dim))
            if self.collapse_fn is collapse_sum:
                res = res * (lens / sublens).unsqueeze(-1).unflatten(-1, (t.ndim - dim % t.ndim)*(1,))
            return self._append_lens(res, lens, dim)
        else:
            return super().forward(t, lens, dim=dim)


@attr.s
class USet(Collapser, BatchedSetModule):
    prenet: Union[Module, Callable[[Tensor], Tensor]]
    postnet: Union[Module, Callable[[tuple[Tensor, Tensor]], Tensor]]
    precollapse: _collapse_fn_t = collapse_mean
    postcollapse: _collapse_fn_t = collapse_mean

    def forward(self, t: Tensor, lens: LongTensor, dim=-2):
        indptr = lens_to_indptr_like(t, lens, dim)
        return self._append_lens(self.postcollapse(self.postnet((
            torch.repeat_interleave(
                self._append_lens(self.precollapse(self.prenet(t), indptr), lens, dim),
                lens, dim=dim
            ),
            t
        )), indptr), lens, dim)


@attr.s
class NestedSetsProcessor(Collapser):
    nets: Iterable[Module]
    collapse_fns: Iterable[_collapse_fn_t] = (collapse_mean,)

    def __attrs_post_init__(self):
        if not isinstance(self.nets, ModuleList):
            self.nets = ModuleList(self.nets)

    def iter_levels(self, lenss: Union[LongTensor, Iterable[LongTensor]]) -> tuple[Iterable[LongTensor], Module, _collapse_fn_t]:
        yield from zip(always_iterable(lenss, Tensor), self.nets, cycle(self.collapse_fns))

    @staticmethod
    def forward_one(t: Tensor, lens: LongTensor, net: Module, collapse_fn: _collapse_fn_t, *args, dim: int) -> Tensor:
        return collapse_fn(net(t), lens_to_indptr_like(t, lens, dim))

    def forward(self, t: Tensor, lenss: Union[LongTensor, Iterable[LongTensor]], dim=-2) -> Tensor:
        return last(
            t for t in [t]
            for lens, *args in self.iter_levels(lenss)
            for t in [self._append_lens(self.forward_one(t, lens, *args, dim=dim), lens, dim)]
        )

    if TYPE_CHECKING:
        __call__ = forward


@attr.s
class SubsampledNestedSetsProcessor(NestedSetsProcessor):
    subsample_counts: Union[Union[int, None], Iterable[Union[int, None]]] = None
    subsample_fracs: Union[float, Iterable[float]] = (1,)

    def iter_levels(self, lenss: Union[LongTensor, Iterable[LongTensor]]) -> tuple[Iterable[LongTensor], Module, _collapse_fn_t, int]:
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
