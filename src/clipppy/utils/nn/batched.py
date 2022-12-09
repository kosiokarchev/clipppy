from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor, Size
from torch.nn import Module, Conv1d, LazyConv1d, MultiheadAttention


def to_batchdim(t: Tensor, event_ndim: int, batch_dim=0):
    return t.unsqueeze(-event_ndim - 1).flatten(end_dim=-event_ndim - 1).movedim(0, batch_dim)


def from_batchdim(t: Tensor, batch_shape: Size, event_ndim: int, batch_dim=0):
    return t.movedim(batch_dim, 0).reshape(batch_shape + t.shape[-event_ndim:])


class BatchedModule(Module):
    event_ndim: ClassVar[int]

    def forward(self, input: Tensor) -> Tensor:
        return from_batchdim(
            super().forward(to_batchdim(input, self.event_ndim)),
            input.shape[:-self.event_ndim], self.event_ndim
        )


class BatchedConv1d(BatchedModule, Conv1d):
    event_ndim = 2


class LazyBatchedConv1d(BatchedConv1d, LazyConv1d):
    cls_to_become = BatchedConv1d


class BatchedMultiheadAttention(MultiheadAttention):
    # TODO: batching of res[1]?

    @property
    def batch_dim(self):
        return 0 if self.batch_first else -2

    def forward(self, query, key, value, *args, **kwargs):
        kw = dict(event_ndim=2, batch_dim=self.batch_dim)
        batch_shape = torch.broadcast_shapes(query.shape[:-2], key.shape[:-2], value.shape[:-2])
        res = super().forward(*(
            to_batchdim(_.expand(batch_shape + _.shape[-2:]), **kw)
            for _ in (query, key, value)
        ), *args, **kwargs)
        return from_batchdim(res[0], batch_shape, **kw), res[1]
