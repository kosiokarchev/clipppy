from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import chain, repeat
from typing import Iterable, TYPE_CHECKING, Callable, NamedTuple, cast

import attr
import torch
from more_itertools import strictly_n, interleave_longest
from torch import Tensor, LongTensor
from torch.nested import nested_tensor_from_jagged
from torch.nested._internal.nested_tensor import NestedTensor
from torch.nn import ModuleList, Module, Parameter, init, LazyLinear, Linear, Sequential, ReLU
from torch.nn.functional import scaled_dot_product_attention

from phytorchx import broadcast_cat
from phytorchx.attrs import AttrsModule, ParametrizedAttrsModule
from ..empty import _empty_module
from ..whiten import LazyWhitenOnline, WhitenOnline
from ... import noop


class SetBatch(NamedTuple):
    val: Tensor
    sizes: LongTensor

    @property
    def indptr(self):
        return broadcast_cat((self.sizes.new_zeros((1,)), self.sizes.cumsum(-1)), -1).to(self.val.device)

    def new(self, val):
        return type(self)(val, self.sizes)

    def to_nested(self):
        return torch.nested.nested_tensor_from_jagged(self.val, lengths=self.sizes)

    @classmethod
    def from_nested(cls, val: NestedTensor):
        sizes = val.lengths()
        if sizes is None:
            sizes = val.offsets().diff()
        return cls(val.values(), sizes)

    def jagged_expand(self, t: Tensor):
        return t.repeat_interleave(self.sizes, -2)

    def std_mean(self):
        mean = self.mean()
        std = self.new(self.val**2).mean().sub_(mean**2).sqrt_()
        return std, mean

    def set_norm(self, eps=1e-6, detach=False):
        std, mean = (self.new(self.val.detach()) if detach else self).std_mean()

        return type(self).from_nested(
            (self.to_nested() - mean.unsqueeze(1)) / (std+eps).unsqueeze(1)
        ), mean, std

    def expand_like(self, other) -> SetBatch:
        x, sizes = self

        if len(self.sizes) == 1 and len(other.shape) > 1:
            sizes = self.sizes.repeat(other.shape[-2])
            x = self.val.expand(other.shape[-2], *self.val.shape).flatten(end_dim=1)
        else:
            assert len(self.sizes) == other.shape[0]

        return type(self)(x, sizes)


    def cat(self, other: Tensor):
        x, sizes = self.expand_like(other)

        return type(self)(broadcast_cat((
            other.expand(*other.shape[:-2], len(sizes), other.shape[-1]).repeat_interleave(sizes, -2),
            x
        ), dim=-1), sizes)

    @property
    def vals(self):
        return self.val.split(self.sizes.tolist(), dim=0)

    def reduce(self, how) -> Tensor:
        from torch_scatter import segment_csr
        return segment_csr(self.val, self.indptr, reduce=how)

    def sum(self):
        return self.reduce('sum')
        # return self.reduce('sum') if self.val.requires_grad else self.to_nested().sum(1)

    def mean(self):
        return self.reduce('mean')
        # return self.reduce('mean') if self.val.requires_grad else self.to_nested().mean(1)


@attr.s(eq=False, auto_attribs=True)
class SetNorm(AttrsModule):
    append: bool = True
    whiten: bool = True
    eps: float = 1e-6

    def __attrs_post_init__(self):
        self.whitener = LazyWhitenOnline() if self.whiten else _empty_module

    def forward(self, batch: SetBatch, *extras: Tensor) -> SetBatch:
        batch, mean, std = batch.set_norm(self.eps)
        return batch.cat(self.whitener(broadcast_cat((mean, std) + extras, -1))) if self.append else batch


@attr.s(eq=False, auto_attribs=True)
class Elementwise(AttrsModule):
    mod: Module

    def forward(self, batch: SetBatch) -> SetBatch:
        return SetBatch(self.mod(batch.val), batch.sizes)


@attr.s(eq=False, auto_attribs=True)
class CrossAttentionLayer(ParametrizedAttrsModule):
    out_features: int
    nheads: int = 1

    key_embedder: Callable[[Tensor], Tensor] = _empty_module
    dropout: float = 0.

    def __attrs_post_init__(self):
        self.value = Parameter(torch.empty(self.out_features, **self.factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.value.unsqueeze(-2))

    @staticmethod
    def _nested_like(ex, other):
        return nested_tensor_from_jagged(other, lengths=ex.new_ones(len(ex), dtype=int))

    def forward(self, key: Tensor, query: NestedTensor) -> NestedTensor:
        return cast(NestedTensor, scaled_dot_product_attention(*(v.unflatten(-1, (self.nheads, -1)).transpose(1, 2) for v in (
            query,
            self._nested_like(key, self.key_embedder(key)),
            self._nested_like(key, self.value.expand(len(key), -1).contiguous())
        )), dropout_p=self.dropout if self.training else 0.).transpose(1, 2).flatten(-2))

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(eq=False, auto_attribs=True)
class CrossEncoder(AttrsModule):
    nfeatures: int | Iterable[int]
    nlayers: int = None
    nheads: int | Iterable[int] = 1

    nfeatures_in: int = attr.ib(init=False)

    def __attrs_post_init__(self):
        if isinstance(self.nfeatures, int):
            self.nfeatures = repeat(self.nfeatures)
        if isinstance(self.nheads, int):
            self.nheads = repeat(self.nheads)

        self.nfeatures_in = next(_nfeatures := iter(self.nfeatures))

        if self.nlayers is None:
            self.nfeatures, self.nheads = zip(*zip(_nfeatures, self.nheads))
        else:
            self.nfeatures = list(strictly_n(_nfeatures, self.nlayers, too_long=noop))
            self.nheads = list(strictly_n(self.nheads, self.nlayers, too_long=noop))

        self.cross_attentions = ModuleList(
            CrossAttentionLayer(nfeatures, nheads, LazyLinear(nfeatures_in))
            for nfeatures_in, nfeatures, nheads in zip(
                chain((self.nfeatures_in,), self.nfeatures),
                self.nfeatures, self.nheads
            )
        )
        self.fcs = ModuleList(Sequential(Linear(n, n), ReLU()) for n in self.nfeatures)

    def forward(self, key: Tensor, query: NestedTensor) -> NestedTensor:
        for mod in interleave_longest(self.cross_attentions, self.fcs):
            query = mod(key, query) if isinstance(mod, CrossAttentionLayer) else query
        return query

    if TYPE_CHECKING:
        __call__ = forward


class BatchedSetModule(Module, ABC):
    @abstractmethod
    def forward(self, batch: SetBatch) -> Tensor: ...

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(eq=False, auto_attribs=True)
class MappedSetModule(BatchedSetModule, AttrsModule):
    net: Module | Callable[[Tensor], Tensor]

    def forward(self, batch: SetBatch) -> Tensor:
        return torch.stack(tuple(map(self.net, batch.vals)), dim=0)


@attr.s(eq=False, auto_attribs=True)
class ForDataParallel(AttrsModule):
    mod: Module
    batch_size: int
    batch_dim: int = 0

    def forward(self, t: Tensor):
        return torch.cat([
            self.mod(_t) for _t in t.split(self.batch_size, dim=self.batch_dim)
        ], self.batch_dim)


# _collapse_fn_t: TypeAlias = Callable[[Tensor, Iterable[int]], Tensor]
#
#
# def _collapse(t: Tensor, indptr: Tensor, reduce: Literal['mean', 'sum']):
#     from torch_scatter import segment_csr
#     return segment_csr(t, indptr, reduce=reduce)
#
#
# collapse_sum: _collapse_fn_t = partial(_collapse, reduce='sum')
# collapse_mean: _collapse_fn_t = partial(_collapse, reduce='mean')
#
#
# def collapse_nmean(t: Tensor, indptr: Tensor) -> Tensor:
#     return torch.mul(*fancy_align(collapse_mean(t, indptr), indptr.diff(n=1, dim=-1).to(t).sqrt_()))


def __getattr__(name):
    from warnings import warn

    if name == '_collapse':
        warn('Collapsing has moved to SetBatch', DeprecationWarning)
        return SetBatch.reduce

    raise AttributeError(name)


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class Collapser(AttrsModule):
    append_lens: bool = True
    whiten_lens: bool = True
    lens_scale: float = None

    def __attrs_post_init__(self):
        self.lens_whitener = WhitenOnline((1,)) if self.whiten_lens else _empty_module

    def _append_lens(self, t: Tensor, lens: LongTensor) -> Tensor:
        if self.append_lens:
            lens = lens.to(t).unsqueeze(-1)

            # TODO: un-hotfix
            if hasattr(self, 'lens_scale') and self.lens_scale:
                lens = lens / self.lens_scale

            return broadcast_cat((t, self.lens_whitener(lens)), -1)
        else:
            return t


@attr.s(eq=False, auto_attribs=True)
class SetCollapser(Collapser, BatchedSetModule):
    net: Module | Callable[[SetBatch], SetBatch] = _empty_module
    # collapse_fn: _collapse_fn_t = collapse_mean
    reduce_fn: Callable[[SetBatch], Tensor] = SetBatch.mean

    def forward(self, batch: SetBatch) -> Tensor:
        return self._append_lens(self.reduce_fn(self.net(batch)), batch.sizes)
        # return self._append_lens(self.collapse_fn(
        #     self.net(batch).val, batch.indptr), batch.sizes)


# @attr.s(eq=False, auto_attribs=True)
# class USet(Collapser, BatchedSetModule):
#     prenet: Module | Callable[[Tensor], Tensor]
#     postnet: Module | Callable[[tuple[Tensor, Tensor]], Tensor]
#     precollapse: _collapse_fn_t = collapse_mean
#     postcollapse: _collapse_fn_t = collapse_mean
#
#     def forward(self, t: Tensor, lens: LongTensor, dim=-2):
#         indptr = lens_to_indptr_like(t, lens, dim)
#         return self._append_lens(self.postcollapse(self.postnet((
#             torch.repeat_interleave(
#                 self._append_lens(self.precollapse(self.prenet(t), indptr), lens, dim),
#                 lens, dim=dim
#             ),
#             t
#         )), indptr), lens, dim)
#
#
# @attr.s(eq=False, auto_attribs=True)
# class NestedSetsProcessor(Collapser):
#     nets: Iterable[Module]
#     collapse_fns: Iterable[_collapse_fn_t] = (collapse_mean,)
#
#     def __attrs_post_init__(self):
#         if not isinstance(self.nets, ModuleList):
#             self.nets = ModuleList(self.nets)
#
#     def iter_levels(self, lenss: LongTensor | Iterable[LongTensor]) -> tuple[Iterable[LongTensor], Module, _collapse_fn_t]:
#         yield from zip(always_iterable(lenss, Tensor), self.nets, cycle(self.collapse_fns))
#
#     @staticmethod
#     def forward_one(t: Tensor, lens: LongTensor, net: Module, collapse_fn: _collapse_fn_t, *args, dim: int) -> Tensor:
#         return collapse_fn(net(t), lens_to_indptr_like(t, lens, dim))
#
#     def forward(self, t: Tensor, lenss: LongTensor | Iterable[LongTensor], dim=-2) -> Tensor:
#         return last(
#             t for t in [t]
#             for lens, *args in self.iter_levels(lenss)
#             for t in [self._append_lens(self.forward_one(t, lens, *args, dim=dim), lens, dim)]
#         )
#
#     if TYPE_CHECKING:
#         __call__ = forward
