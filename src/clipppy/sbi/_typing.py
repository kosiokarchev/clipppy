from __future__ import annotations

from typing import Iterable, Mapping, Protocol, TypeVar

from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import TypeAlias

from .data import SBIDataLoader, SBIDataset
from .nn import BaseSBIHead, MultiSBITail


class MultiSBIProtocol(Protocol):
    param_names: Iterable[str]
    obs_names: Iterable[str]

    loader: SBIDataLoader
    dataset: SBIDataset

    head: BaseSBIHead
    tail: MultiSBITail


_OptimizerT = TypeVar('_OptimizerT', bound=Optimizer)
_SchedulerT = TypeVar('_SchedulerT')
_SBIBatchT: TypeAlias = tuple[Mapping[str, Tensor], Mapping[str, Tensor]]
DEFAULT_LOSS_NAME = 'loss'
