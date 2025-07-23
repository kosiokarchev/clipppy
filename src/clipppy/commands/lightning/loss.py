from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import starmap
from numbers import Number
from typing import Any, Generic, Iterable, Literal, Mapping, NamedTuple, TYPE_CHECKING, TypeVar, Callable

import torch
from more_itertools import all_equal
from torch import Tensor
from torch.nn.functional import logsigmoid
from torch.utils._pytree import _broadcast_to_and_flatten, tree_flatten, tree_unflatten, TreeSpec
from typing_extensions import ParamSpec, Self, TypeAlias

from ...sbi._typing import _Tree, _KT
from ...sbi.nn.npe import NPEResult
from ...utils import Sentinel

_Tin = TypeVar('_Tin')
_DimT: TypeAlias = int | tuple[int, ...] | tuple[str, ...] | Literal[Sentinel.skip]
_DimTreeT: TypeAlias = _DimT | Iterable['_DimTreeT'] | Mapping[Any, '_DimTreeT']
_ReduceFuncT: TypeAlias = Callable[[Tensor, ...], Tensor]
_ReduceFuncTreeT: TypeAlias = _ReduceFuncT | Iterable['_ReduceFuncT'] | Mapping[Any, '_ReduceFuncT']

_t: TypeAlias = Tensor | Number


_LossParamsT = ParamSpec('_LossParamsT')


class BaseSBILoss(Generic[_LossParamsT]):
    class ReturnT(NamedTuple):
        loss: Tensor
        flat: list[Tensor] = None
        spec: TreeSpec = None

        @classmethod
        def from_mapping(cls, rts: Mapping[_KT, Self]) -> Self:
            return cls(
                sum(r.loss for r in rts.values()),
                *tree_flatten({key: r.unflatten() for key, r in rts.items()})
            )

        def tree_binary_op(self, op: Callable[[Tensor, _Tin], Tensor], other: _Tree) -> Self:
            newflat = list(starmap(op, zip(self.flat, _broadcast_to_and_flatten(other, self.spec))))
            return type(self)(sum(newflat), newflat, self.spec)

        def __mul__(self, other) -> Self:
            return type(self)(other * self.loss, [other * f for f in self.flat], self.spec)

        __rmul__ = __mul__

        def unflatten(self):
            return tree_unflatten(self.flat, self.spec)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> BaseSBILoss.ReturnT: ...


@dataclass
class MultiLoss(BaseSBILoss[_LossParamsT], Generic[_LossParamsT]):
    losses: Mapping[str, BaseSBILoss]
    weights: Mapping[str, Number | Tensor] = field(default_factory=dict)

    def __call__(self, *args, **kwargs) -> BaseSBILoss.ReturnT:
        return self.ReturnT.from_mapping({
            key: l if w is None else w*l
            for key, loss in self.losses.items()
            for w, l in [(self.weights.get(key, None), loss(*args, **kwargs))]
        })


@dataclass
class SBILoss(BaseSBILoss[_LossParamsT], Generic[_LossParamsT], ABC):
    dim: _DimTreeT = Sentinel.empty
    reduce_func: _ReduceFuncTreeT = torch.mean

    @staticmethod
    def _reduce(loss: Tensor, dim: _DimT, reduce_func: _ReduceFuncT):
        return (loss if dim is Sentinel.skip else reduce_func(loss, *(
            (dim,) if dim is not Sentinel.empty else ()
        )))

    @abstractmethod
    def _loss(self, *args: _LossParamsT.args, **kwargs: _LossParamsT.kwargs): ...

    def _call_one(self, *args: _LossParamsT.args, dim: _DimT, reduce_func: _ReduceFuncT, **kwargs: _LossParamsT.kwargs) -> Tensor:
        return self._reduce(self._loss(*args, **kwargs), dim, reduce_func)

    def _call(self, flat: Iterable[_LossParamsT.args], spec: TreeSpec):
        return self.ReturnT(sum(res := [
            self._call_one(*args, dim=dim, reduce_func=reduce_func)
            for args, dim, reduce_func in zip(flat, _broadcast_to_and_flatten(self.dim, spec), _broadcast_to_and_flatten(self.reduce_func, spec))
        ]) / len(res), res, spec)

    def __call__(self, *args: _Tree):
        flats, specs = zip(*map(tree_flatten, args))
        assert all_equal(specs)
        return self._call(zip(*flats), specs[0])


class NPELoss(SBILoss):
    def _loss(self, nperes: NPEResult, *args):
        return - nperes.log_prob

    if TYPE_CHECKING:
        def __call__(self, nperes: _Tree[NPEResult], *args: _Tree) -> BaseSBILoss.ReturnT: ...


class BaseNRELoss(SBILoss, ABC):
    if TYPE_CHECKING:
        def __call__(
            self, log_ratio_joint: _Tree[Tensor], log_ratio_marginal: _Tree[Tensor],
            weight_joint: _Tree[_t] = 1., weight_marginal: _Tree[_t] = 1.
        ) -> BaseSBILoss.ReturnT: ...


class BCENRELoss(BaseNRELoss):
    def _loss(self, log_ratio_joint: Tensor, log_ratio_marginal: Tensor,
              weight_joint: _t = 1., weight_marginal: _t = 1.):
        return - (weight_joint * logsigmoid(log_ratio_joint) +
                  weight_marginal * logsigmoid(-log_ratio_marginal))


class LogisticNRELoss(BaseNRELoss):
    def _loss(self, log_ratio_joint: Tensor, log_ratio_marginal: Tensor,
              weight_joint: _t = 1., weight_marginal: _t = 1.):
        return (weight_joint * torch.logaddexp(-log_ratio_joint, log_ratio_joint.new_zeros(())) +
                weight_marginal * torch.logaddexp(log_ratio_marginal, log_ratio_marginal.new_zeros(())))


class SavageNRELoss(BaseNRELoss):
    def _loss(self, log_ratio_joint: Tensor, log_ratio_marginal: Tensor,
              weight_joint: _t = 1., weight_marginal: _t = 1.):
        return (weight_joint / (1+log_ratio_joint.exp()).square_() +
                weight_marginal / (1+(-log_ratio_marginal).exp()).square_())


class ExpNRELoss(BaseNRELoss):
    def _loss(self, log_ratio_joint: Tensor, log_ratio_marginal: Tensor,
              weight_joint: _t = 1., weight_marginal: _t = 1.):
        return (weight_joint / (log_ratio_joint / 2).exp_() +
                weight_marginal * (log_ratio_marginal / 2).exp_())
