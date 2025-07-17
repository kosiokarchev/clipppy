from __future__ import annotations

from typing import get_type_hints, Sequence, Any, ClassVar, Union

import attr
import torch
import torch.distributed as distrib
from torch import Tensor, Size
from torch.nn import init, UninitializedBuffer, SyncBatchNorm, UninitializedParameter, Parameter
from torch.nn.modules.lazy import LazyModuleMixin

from phytorchx.attrs import ParametrizedAttrsModule
from ..typing import _Tensor_like


@attr.s(eq=False, auto_attribs=True)
class Whiten(ParametrizedAttrsModule):
    shape: Union[Size, tuple[int, ...], list[int]]

    affine: bool = attr.ib(default=False, kw_only=True)
    momentum: float = attr.ib(default=0, kw_only=True)

    _buffer_names: ClassVar = 'mean', 'std'
    mean: Tensor = attr.ib(init=False, repr=False)
    std: Tensor = attr.ib(init=False, repr=False)

    _param_names: ClassVar = 'weight', 'bias'
    weight: Tensor = attr.ib(init=False, repr=False)
    bias: Tensor = attr.ib(init=False, repr=False)


    def __attrs_post_init__(self):
        self.ndim = len(self.shape)

        for name in self._buffer_names:
            self.register_buffer(name, torch.empty(self.shape, **self.factory_kwargs))

        if self.affine:
            for name in self._param_names:
                self.register_parameter(name, Parameter(torch.empty(self.shape, **self.factory_kwargs)))

        self.reset_()

    def reset_(self):
        for name in self._buffer_names:
            init.zeros_(getattr(self, name))
        if self.affine:
            init.zeros_(self.bias)
            init.ones_(self.weight)

    def _whiten_list(self, tensors: Sequence[Tensor]) -> list[Tensor]:
        n = len(tensors)
        res = torch._foreach_sub(tensors, n*(self.mean,))
        torch._foreach_div_(res, n*(self.std,))
        if self.affine:
            torch._foreach_mul(res, n*(self.weight,))
            torch._foreach_add(res, n*(self.bias,))
        return list(res)

    def forward(self, a: _Tensor_like):
        if not torch.is_tensor(a):
            return self._whiten_list(a)
        elif a.is_nested:
            return torch.nested.as_nested_tensor(self._whiten_list(a.unbind()))
        else:
            res = (a - self.mean).divide_(self.std)
            if self.affine:
                res.mul_(self.weight).add_(self.bias)
            return res


@attr.s(eq=False)
class WhitenOnline(Whiten):
    _buffer_names: ClassVar = Whiten._buffer_names + ('mean_square',)
    mean_square: Tensor = attr.ib(init=False, repr=False)

    n: int = attr.ib(init=False)
    group = attr.ib(default=None, kw_only=True)

    def reset_(self):
        self.n = 0
        super().reset_()

    def get_extra_state(self) -> Any:
        return dict(n=self.n)

    def set_extra_state(self, state: Any):
        self.n = state.get('n', 0)

    def _update_(self, a: Tensor):
        n = len(a)
        nnew = self.n + n
        wnew = 1 - self.n/nnew * (1-self.momentum)

        a = a.to(self.mean)

        self.mean.lerp_(a.mean(0), wnew)
        self.mean_square.lerp_(torch.linalg.vector_norm(a, dim=0).pow_(2).div_(n), wnew)

        if distrib.is_available() and distrib.is_initialized():
            distrib.all_reduce(self.mean, distrib.ReduceOp.AVG, self.group)
            distrib.all_reduce(self.mean_square, distrib.ReduceOp.AVG, self.group)

        self.std = (self.mean_square - self.mean**2).clamp_(1e-12).sqrt_()
        self.n = nnew

    def forward(self, a: Tensor):
        if self.training:
            with torch.no_grad():
                if a.is_nested:
                    _a = torch.Tensor(a.storage()).reshape(-1, *self.shape)
                else:
                    _a = a.unsqueeze(0).flatten(end_dim=a.ndim-self.ndim)

                self._update_(_a)

        return a if self.n < 2 else super().forward(a)

    def freeze_(self):
        self.__class__ = Whiten
        return self


class LazyWhitenOnline(LazyModuleMixin, WhitenOnline):
    cls_to_become = WhitenOnline

    mean: UninitializedBuffer
    mean_square: UninitializedBuffer
    std: UninitializedBuffer

    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(self, ndim=1, affine=False, momentum: float = 0, group=None, device=None, dtype=None):
        super().__init__(shape=(), affine=affine, momentum=momentum, group=group, device=device, dtype=dtype)
        self.ndim = ndim
        for name in self._buffer_names:
            setattr(self, name, UninitializedBuffer())
        if self.affine:
            for name in self._param_names:
                setattr(self, name, UninitializedParameter())

    def reset_(self):
        if not self.has_uninitialized_params():
            super().reset_()

    def initialize_parameters(self, a: Tensor):
        if self.has_uninitialized_params():
            self.shape = torch.Size(a.size(i) for i in range(a.ndim-self.ndim, a.ndim))
            self.device = a.device
            self.dtype = a.dtype

            with torch.no_grad():
                for name in self._buffer_names + (self._param_names if self.affine else ()):
                    getattr(self, name).materialize(shape=self.shape, **self.factory_kwargs)
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
