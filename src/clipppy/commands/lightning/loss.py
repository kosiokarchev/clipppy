from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import Any, Iterable, Literal, Mapping, NamedTuple, TypeVar, Union

from torch import Tensor
from torch.nn.functional import logsigmoid
from torch.utils._pytree import _broadcast_to_and_flatten, tree_flatten, TreeSpec
from typing_extensions import TypeAlias

from ...utils import Sentinel


_Tree: TypeAlias = Union[Tensor, Iterable['_Tree'], Mapping[Any, '_Tree']]
_TreeT = TypeVar('_TreeT', bound=_Tree)

_DimT: TypeAlias = Union[int, tuple[int, ...], tuple[str, ...], Literal[Sentinel.skip]]
_DimTreeT: TypeAlias = Union[_DimT, Iterable['_DimTreeT'], Mapping[Any, '_DimTreeT']]


@dataclass
class NRELoss:
    class ReturnT(NamedTuple):
        loss: Tensor
        flat: list[Tensor] = None
        spec: TreeSpec = None

    dim: _DimTreeT = Sentinel.empty

    @staticmethod
    def _call_one(log_ratio_joint: Tensor, log_ratio_marginal: Tensor, dim: _DimT) -> Tensor:
        res = logsigmoid(log_ratio_joint) + logsigmoid(-log_ratio_marginal)
        return -(res if dim is Sentinel.skip else res.mean(*(
            (dim,) if dim is not Sentinel.empty else ()
        )))

    def __call__(self, log_ratio_joint: _TreeT, log_ratio_marginal: _TreeT) -> ReturnT:
        (flat_joint, spec), (flat_marginal, spec_marginal) = map(tree_flatten, (log_ratio_joint, log_ratio_marginal))
        assert spec == spec_marginal
        res = list(starmap(self._call_one, zip(
            flat_joint, flat_marginal, _broadcast_to_and_flatten(self.dim, spec)
        )))
        return self.ReturnT(sum(res) / len(res), res, spec)
