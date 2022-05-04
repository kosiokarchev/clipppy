from typing import Iterable, Sequence

import torch.optim.lr_scheduler as lrs

from . import BaseSchedulerConfig, SchedulerConfig
from ...nre._typing import _OptimizerT, _SchedulerT


def LambdaLR(
    lr_lambda,
    last_epoch=-1, verbose=False,
    interval: 'str' = 'step', frequency: 'int' = 1,
    monitor: 'str' = 'loss', strict: 'bool' = True, name: 'str' = None
) -> SchedulerConfig[lrs.LambdaLR]: ...

def MultiplicativeLR(
    lr_lambda,
    last_epoch=-1, verbose=False,
    interval: 'str' = 'step', frequency: 'int' = 1,
    monitor: 'str' = 'loss', strict: 'bool' = True, name: 'str' = None
) -> SchedulerConfig[lrs.MultiplicativeLR]: ...

def StepLR(
    step_size, gamma=0.1,
    last_epoch=-1, verbose=False,
    interval: 'str' = 'step', frequency: 'int' = 1,
    monitor: 'str' = 'loss', strict: 'bool' = True, name: 'str' = None
) -> SchedulerConfig[lrs.StepLR]: ...

def MultiStepLR(
    milestones, gamma=0.1,
    last_epoch=-1, verbose=False,
    interval: 'str' = 'step', frequency: 'int' = 1,
    monitor: 'str' = 'loss', strict: 'bool' = True, name: 'str' = None
) -> SchedulerConfig[lrs.MultiStepLR]: ...

def ConstantLR(
    factor=1/3, total_iters=5,
    last_epoch=-1, verbose=False,
    interval: 'str' = 'step', frequency: 'int' = 1,
    monitor: 'str' = 'loss', strict: 'bool' = True, name: 'str' = None
) -> SchedulerConfig[lrs.ConstantLR]: ...

def LinearLR(
    start_factor=1/3, end_factor=1.0, total_iters=5,
    last_epoch=-1, verbose=False,
    interval: 'str' = 'step', frequency: 'int' = 1,
    monitor: 'str' = 'loss', strict: 'bool' = True, name: 'str' = None
) -> SchedulerConfig[lrs.LinearLR]: ...

def ExponentialLR(
    gamma,
    last_epoch=-1, verbose=False,
    interval: 'str' = 'step', frequency: 'int' = 1,
    monitor: 'str' = 'loss', strict: 'bool' = True, name: 'str' = None
) -> SchedulerConfig[lrs.ExponentialLR]: ...

def CosineAnnealingLR(
    T_max, eta_min=0,
    last_epoch=-1, verbose=False,
    interval: 'str' = 'step', frequency: 'int' = 1,
    monitor: 'str' = 'loss', strict: 'bool' = True, name: 'str' = None
) -> SchedulerConfig[lrs.CosineAnnealingLR]: ...

def CosineAnnealingWarmRestarts(
    T_0, T_mult=1, eta_min=0,
    last_epoch=-1, verbose=False,
    interval: 'str' = 'step', frequency: 'int' = 1,
    monitor: 'str' = 'loss', strict: 'bool' = True, name: 'str' = None
) -> SchedulerConfig[lrs.CosineAnnealingWarmRestarts]: ...

def CyclicLR(
    base_lr, max_lr, step_size_up=2000, step_size_down=None,
    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
    cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
    last_epoch=-1, verbose=False,
    interval: 'str' = 'step', frequency: 'int' = 1,
    monitor: 'str' = 'loss', strict: 'bool' = True, name: 'str' = None
) -> SchedulerConfig[lrs.CyclicLR]: ...

def OneCycleLR(
    max_lr, total_steps=None, epochs=None, steps_per_epoch=None,
    pct_start=0.3, anneal_strategy='cos',
    cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
    div_factor=25.0, final_div_factor=10000.0, three_phase=False,
    last_epoch=-1, verbose=False,
    interval: 'str' = 'step', frequency: 'int' = 1,
    monitor: 'str' = 'loss', strict: 'bool' = True, name: 'str' = None
) -> SchedulerConfig[lrs.OneCycleLR]: ...


class BaseSchedulerCollectionConfig(BaseSchedulerConfig[_SchedulerT]):
    scheduler_configs: Iterable[BaseSchedulerConfig]
    def _schedulers(self, optimizer: _OptimizerT, **kwargs) -> list[_OptimizerT]: ...
    def __init__(self, scheduler_configs) -> None: ...

class ChainedScheduler(BaseSchedulerCollectionConfig[lrs.ChainedScheduler]):
    def _scheduler(self, optimizer: _OptimizerT, **kwargs): ...

class SequentialLR(BaseSchedulerCollectionConfig[lrs.SequentialLR]):
    milestones: Sequence[int]
    last_epoch: int
    def _scheduler(self, optimizer: _OptimizerT, **kwargs): ...
    def __init__(self, scheduler_configs, milestones, last_epoch) -> None: ...
