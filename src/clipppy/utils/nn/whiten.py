from __future__ import annotations

from typing import get_type_hints, Sequence, Any

import torch
from torch import Tensor, Size
from torch.nn import Module, init, UninitializedBuffer, SyncBatchNorm, UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin

from ..typing import _Tensor_like


class WhitenOnline(Module):
    ndim: int

    n: int
    mean: Tensor
    mean_square: Tensor
    std: Tensor

    _buffer_names = 'mean', 'mean_square', 'std'

    def __init__(self, shape=Size(), device=None, dtype=None):
        super().__init__()

        self.ndim = len(shape)

        for name in self._buffer_names:
            self.register_buffer(name, torch.empty(shape, device=device, dtype=dtype))
        self.reset_()

        self.frozen = False

    def reset_(self):
        self.n = 0
        for name in self._buffer_names:
            init.zeros_(getattr(self, name))

    def get_extra_state(self) -> Any:
        return dict(n=self.n)

    def set_extra_state(self, state: Any):
        self.n = state.get('n', 0)

    @property
    def shape(self):
        return self.mean.shape

    def _update_(self, a: Tensor):
        n = len(a)
        nnew = self.n + n
        wold, wnew = self.n/nnew, n/nnew

        self.mean = a.sum(0).div_(nnew).add_(self.mean, alpha=wold)
        self.mean_square = torch.linalg.vector_norm(a, dim=0).pow_(2).div_(nnew).add_(self.mean_square, alpha=wold)
        self.std = (self.mean_square - self.mean**2).clamp_(1e-12).sqrt_()
        self.n = nnew

    def _whiten_list(self, tensors: Sequence[Tensor]) -> list[Tensor]:
        n = len(tensors)
        res = torch._foreach_sub(tensors, n*(self.mean,))
        torch._foreach_div_(res, n*(self.std,))
        return list(res)

    def _whiten(self, a: _Tensor_like):
        if not torch.is_tensor(a):
            return self._whiten_list(a)
        elif a.is_nested:
            return torch.nested.as_nested_tensor(self._whiten_list(a.unbind()))
        else:
            return (a - self.mean).divide_(self.std)

    def forward(self, a: Tensor):
        if self.training and not self.frozen:
            with torch.no_grad():
                if a.is_nested:
                    _a = torch.Tensor(a.storage()).reshape(-1, *self.shape)
                else:
                    _a = a.unsqueeze(0).flatten(end_dim=a.ndim-self.ndim)

                self._update_(_a)

        return a if self.n < 2 else self._whiten(a)


class LazyWhitenOnline(LazyModuleMixin, WhitenOnline):
    cls_to_become = WhitenOnline

    mean: UninitializedBuffer
    mean_square: UninitializedBuffer
    std: UninitializedBuffer

    def __init__(self, ndim=1):
        super().__init__()
        self.ndim = ndim
        for name in self._buffer_names:
            setattr(self, name, UninitializedBuffer())

    def reset_(self):
        if not self.has_uninitialized_params():
            super().reset_()

    def initialize_parameters(self, a: Tensor):
        if self.has_uninitialized_params():
            with torch.no_grad():
                for name in self._buffer_names:
                    getattr(self, name).materialize(
                        # Support NestedTensor
                        shape=torch.Size(a.size(i) for i in range(a.ndim-self.ndim, a.ndim)),
                        device=a.device, dtype=a.dtype
                    )
                self.reset_()


class BatchedSyncBatchNorm(SyncBatchNorm):
    def __init__(self, num_features, ndim=1, **kwargs):
        self.ndim = ndim
        super().__init__(num_features, **kwargs)

    def _coerce_shape(self, t: Tensor):
        return t.flatten(end_dim=t.ndim-self.ndim-1).unsqueeze(-1).flatten(1)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(self._coerce_shape(input)).reshape_as(input)


class LazyBatchedSyncBatchNorm(LazyModuleMixin, BatchedSyncBatchNorm):
    cls_to_become = BatchedSyncBatchNorm

    running_mean: UninitializedBuffer
    running_var: UninitializedBuffer

    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(self, ndim=1, **kwargs):
        super().__init__(num_features=0, ndim=ndim, **kwargs)
        self._lazy_names = (
           ('running_mean', 'running_var') if self.track_running_stats else ()
        ) + (('weight', 'bias') if self.affine else ())

        _utypes = get_type_hints(type(self))
        for name in self._lazy_names:
            setattr(self, name, _utypes[name]())

    def reset_parameters(self):
        if not self.has_uninitialized_params():
            super().reset_parameters()

    def initialize_parameters(self, input: Tensor):
        if self.has_uninitialized_params():
            self.num_features = self._coerce_shape(input).shape[-1]

            with torch.no_grad():
                for name in self._lazy_names:
                    getattr(self, name).materialize(
                        shape=(self.num_features,),
                        device=input.device, dtype=input.dtype
                    )
                self.reset_parameters()
