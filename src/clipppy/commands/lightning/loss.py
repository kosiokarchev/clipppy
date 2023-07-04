from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import starmap
from numbers import Number
from typing import Any, Generic, Iterable, Literal, Mapping, NamedTuple, TYPE_CHECKING, Union, TypeVar, Callable

import torch
from more_itertools import all_equal
from torch import Tensor
from torch.nn.functional import logsigmoid
from torch.utils._pytree import _broadcast_to_and_flatten, tree_flatten, tree_unflatten, TreeSpec
from typing_extensions import ParamSpec, Self, TypeAlias

from ..npe.nn import NPEResult
from ...sbi._typing import _Tree
from ...utils import Sentinel


_Tin = TypeVar('_Tin')
_DimT: TypeAlias = Union[int, tuple[int, ...], tuple[str, ...], Literal[Sentinel.skip]]
_DimTreeT: TypeAlias = Union[_DimT, Iterable['_DimTreeT'], Mapping[Any, '_DimTreeT']]


_LossParamsT = ParamSpec('_LossParamsT')


class BaseSBILoss(Generic[_LossParamsT]):
    class ReturnT(NamedTuple):
        loss: Tensor
        flat: list[Tensor] = None
        spec: TreeSpec = None

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
    weights: Mapping[str, Union[Number, Tensor]] = field(default_factory=dict)

    def __call__(self, *args, **kwargs) -> BaseSBILoss.ReturnT:
        losses: Mapping[str, BaseSBILoss.ReturnT] = {
            key: l if w is None else w*l
            for key, loss in self.losses.items()
            for w, l in [(self.weights.get(key, None), loss(*args, **kwargs))]
        }
        return self.ReturnT(
            sum(r.loss for r in losses.values()),
            *tree_flatten({key: r.unflatten() for key, r in losses.items()}),
        )


@dataclass
class SBILoss(BaseSBILoss[_LossParamsT], Generic[_LossParamsT], ABC):
    dim: _DimTreeT = Sentinel.empty

    @staticmethod
    def _reduce(loss: Tensor, dim: _DimT):
        return (loss if dim is Sentinel.skip else loss.mean(*(
            (dim,) if dim is not Sentinel.empty else ()
        )))

    @abstractmethod
    def _loss(self, *args: _LossParamsT.args, **kwargs: _LossParamsT.kwargs): ...

    def _call_one(self, *args: _LossParamsT.args, dim: _DimT, **kwargs: _LossParamsT.kwargs) -> Tensor:
        return self._reduce(self._loss(*args, **kwargs), dim)

    def _call(self, flat: Iterable[_LossParamsT.args], spec: TreeSpec):
        return self.ReturnT(sum(res := [
            self._call_one(*args, dim=dim)
            for args, dim in zip(flat, _broadcast_to_and_flatten(self.dim, spec))
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


class GANPELoss(SBILoss):
    def _loss(self, nperes: NPEResult, sim_log_prob_grad: Tensor):
        return torch.linalg.vector_norm(nperes.log_prob_grad - sim_log_prob_grad, dim=-1)

    if TYPE_CHECKING:
        def __call__(self, nperes: _Tree[NPEResult], sim_log_prob_grad: _Tree[Tensor]) -> BaseSBILoss.ReturnT: ...


class NRELoss(SBILoss):
    def _loss(self, log_ratio_joint: Tensor, log_ratio_marginal: Tensor,
              weight_joint: Union[Tensor, Number] = 1., weight_marginal: Union[Tensor, Number] = 1.):
        return - (weight_joint * logsigmoid(log_ratio_joint) +
                  weight_marginal * logsigmoid(-log_ratio_marginal))

    if TYPE_CHECKING:
        def __call__(self, log_ratio_joint: _Tree[Tensor], log_ratio_marginal: _Tree[Tensor]) -> BaseSBILoss.ReturnT: ...
                     # weight_joint: _Tree[Union[Tensor, Number]] = 1., weight_marginal: _Tree[Union[Tensor, Number]] = 1.
