from __future__ import annotations

from typing import Iterable, Mapping, Protocol, TypeVar

from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import TypeAlias

from .data import NREDataLoader, NREDataset
from .nn import BaseNREHead, MultiNRETail


class MultiNREProtocol(Protocol):
    param_names: Iterable[str]
    obs_names: Iterable[str]

    loader: NREDataLoader
    dataset: NREDataset

    head: BaseNREHead
    tail: MultiNRETail


_OptimizerT = TypeVar('_OptimizerT', bound=Optimizer)
_SchedulerT = TypeVar('_SchedulerT')
_BatchT: TypeAlias = tuple[Mapping[str, Tensor], Mapping[str, Tensor]]
DEFAULT_LOSS_NAME = 'loss'
