from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any, Mapping, Type, Union

import torch.optim
from frozendict import frozendict
from torch.utils.tensorboard import SummaryWriter

from .optimizing_command import OptimizingCommand


class Callback:
    def __call__(self, command: OptimizingCommand, i, loss, args, kwargs):
        pass


class TensorboardCallback(Callback, SummaryWriter):
    _loss_tag = 'loss'
    _optim_tag = 'optim'

    def __init__(self, log_dir: Union[str, Path], max_queue=1000, flush_secs=10,
                 i0: int = 0, suffix='',
                 constant_scalars: Mapping[str, Any] = frozendict()):
        super().__init__(str(log_dir), max_queue=max_queue, flush_secs=flush_secs)
        self.i0 = i0
        self.suffix = suffix
        self.constant_scalars = constant_scalars

    def _suffixed(self, tag: str):
        return '/'.join(filter(bool, (tag, self.suffix)))

    def global_step(self, i: int):
        return self.i0 + i

    def __call__(self, command: OptimizingCommand, i, loss, args, kwargs):
        global_step = self.global_step(i)

        for key, val in chain(self.constant_scalars.items(), ((self._loss_tag, loss),)):
            self.add_scalar(self._suffixed(key), val, global_step)

        if 'optim' in kwargs and isinstance(o := kwargs['optim'], torch.optim.Optimizer):
            for pgi, pg in enumerate(o.param_groups):
                self.add_scalar(f'{self._optim_tag}/lr/param_group_{pgi}', pg['lr'], global_step)
            # self.add_scalars(self._optim_tag, {
            #     self._suffixed(f'lr/param_group_{pgi}'): pg['lr']
            #     for pgi, pg in enumerate(o.param_groups)
            # }, global_step)


class LRSchedulerCallback(Callback):
    def __init__(self, scheduler=None, scheduler_cls: Type = None, takes_loss=False, **scheduler_kwargs):
        self.scheduler = scheduler
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs

        self.takes_loss = takes_loss

    def __call__(self, command: OptimizingCommand, i, loss, args, kwargs):
        if self.scheduler is None:
            self.scheduler = self.scheduler_cls(kwargs['optim'], **self.scheduler_kwargs)
        self.scheduler.step(*self.takes_loss*(loss,))
