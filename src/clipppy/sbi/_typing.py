from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Iterable, Mapping, Protocol, TypeVar, Generic, Any, Union

from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import TypeAlias


_KT = TypeVar('_KT')
_MultiKT = Union[_KT, Iterable[_KT]]


from . import data, nn


class MultiSBIProtocol(Protocol):
    param_names: Iterable[str]
    obs_names: Iterable[str]

    loader: data.SBIDataLoader
    dataset: data.SBIDataset

    head: nn.BaseSBIHead
    tail: nn.MultiSBITail


_TreeV = TypeVar('_TreeV')
_Tree: TypeAlias = Union[_TreeV, Iterable['_Tree'], Mapping[Any, '_Tree']]
_MappingT = TypeVar('_MappingT', bound=Mapping[str, Tensor])
_SBIParamsT: TypeAlias = _MappingT
_SBIObsT: TypeAlias = _MappingT
_SBIWeightT: TypeAlias = _Tree[Union[Tensor, Number]]


@dataclass
class SBIBatch(Generic[_MappingT]):
    params: _SBIParamsT
    obs: _SBIObsT
    weight: _SBIWeightT = 1


_OptimizerT = TypeVar('_OptimizerT', bound=Optimizer)
_SchedulerT = TypeVar('_SchedulerT')
DEFAULT_LOSS_NAME = 'loss'
DEFAULT_VAL_NAME = 'val'
