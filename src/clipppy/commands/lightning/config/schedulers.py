from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Sequence

import torch.optim.lr_scheduler as lrs

from . import BaseSchedulerConfig, SchedulerConfig
from ....sbi._typing import _OptimizerT, _SchedulerT


_og_lrs = (
    lrs.LambdaLR, lrs.MultiplicativeLR, lrs.StepLR, lrs.MultiStepLR,
    lrs.ConstantLR, lrs.LinearLR, lrs.ExponentialLR,
    lrs.CosineAnnealingLR, lrs.CosineAnnealingWarmRestarts,
    lrs.CyclicLR, lrs.OneCycleLR,
)

__all__ = *(cls.__name__ for cls in _og_lrs), 'ChainedScheduler', 'SequentialLR'


for lr_scheduler in _og_lrs:
    globals()[lr_scheduler.__name__] = SchedulerConfig.wrap(lr_scheduler)


@dataclass
class BaseSchedulerCollectionConfig(BaseSchedulerConfig[_SchedulerT], Generic[_SchedulerT]):
    scheduler_configs: Iterable[BaseSchedulerConfig] = ()

    def _schedulers(self, optimizer: _OptimizerT, **kwargs) -> list[_OptimizerT]:
        return [sc._scheduler(optimizer, **kwargs) for sc in self.scheduler_configs]


class ChainedScheduler(BaseSchedulerCollectionConfig[lrs.ChainedScheduler]):
    def _scheduler(self, optimizer: _OptimizerT, **kwargs):
        return lrs.ChainedScheduler(self._schedulers(optimizer, **kwargs))


@dataclass
class SequentialLR(BaseSchedulerCollectionConfig[lrs.SequentialLR]):
    milestones: Sequence[int] = ()
    last_epoch: int = -1

    def _scheduler(self, optimizer: _OptimizerT, **kwargs):
        return lrs.SequentialLR(
            optimizer=optimizer,
            schedulers=self._schedulers(optimizer, **kwargs),
            milestones=self.milestones, last_epoch=self.last_epoch)
