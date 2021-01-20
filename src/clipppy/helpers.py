import typing

import pyro.optim
import torch

from .globals import register_globals
from .utils import dict_union

__all__ = ('scheduled_optimizer',)


def scheduled_optimizer(lr_scheduler_cls: typing.Type[pyro.optim.PyroLRScheduler],
                        optimizer_cls: typing.Type[torch.optim.lr_scheduler._LRScheduler],
                        clip_args=None, **scheduler_kwargs):
    scheduler_kwargs = dict_union({'verbose': True}, scheduler_kwargs)
    def _f(optim_args):
        return pyro.optim.PyroLRScheduler(
            lr_scheduler_cls, {'optimizer': optimizer_cls, 'optim_args': optim_args, **scheduler_kwargs},
            clip_args=clip_args
        )
    return _f


register_globals(**{a: globals()[a] for a in __all__ if a in globals()})
