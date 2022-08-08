from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Iterable, Literal, Mapping, NamedTuple, TypeVar, Union

from torch import Tensor
from torch.nn.functional import logsigmoid
from torch.utils._pytree import _broadcast_to_and_flatten, tree_flatten, TreeSpec
from typing_extensions import ParamSpec, TypeAlias

from ..npe._typing import Distribution
from ...utils import Sentinel


_Tree: TypeAlias = Union[Tensor, Iterable['_Tree'], Mapping[Any, '_Tree']]
_TreeT = TypeVar('_TreeT', bound=_Tree)

_DimT: TypeAlias = Union[int, tuple[int, ...], tuple[str, ...], Literal[Sentinel.skip]]
_DimTreeT: TypeAlias = Union[_DimT, Iterable['_DimTreeT'], Mapping[Any, '_DimTreeT']]


_LossParamsT = ParamSpec('_LossParamsT')


@dataclass
class SBILoss(Generic[_LossParamsT]):
    class ReturnT(NamedTuple):
        loss: Tensor
        flat: list[Tensor] = None
        spec: TreeSpec = None

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

    @abstractmethod
    def __call__(self, *args, **kwargs) -> SBILoss.ReturnT: ...


class MultiNPELoss(SBILoss):
    def _loss(self, theta: Tensor, q: Distribution):
        return - q.log_prob(theta)

    def __call__(self, nperes: _TreeT):
        return self._call(*tree_flatten(nperes))


class MultiNRELoss(SBILoss):
    def _loss(self, log_ratio_joint: Tensor, log_ratio_marginal: Tensor):
        return - (logsigmoid(log_ratio_joint) + logsigmoid(-log_ratio_marginal))

    def __call__(self, log_ratio_joint: _TreeT, log_ratio_marginal: _TreeT):
        (flat_joint, spec), (flat_marginal, spec_marginal) = map(tree_flatten, (log_ratio_joint, log_ratio_marginal))
        assert spec == spec_marginal
        return self._call(zip(flat_joint, flat_marginal), spec)
