import typing

import pyro.optim
import torch

from .utils import dict_union


__all__ = 'scheduled_optimizer', 'scheduled_optimizer_callback', 'scheduled_optimizer_callback_with_loss'


def scheduled_optimizer(lr_scheduler_cls: typing.Type[pyro.optim.PyroLRScheduler],
                        optimizer_cls: typing.Type[torch.optim.lr_scheduler._LRScheduler],
                        clip_args=None, **scheduler_kwargs):
    scheduler_kwargs = dict_union({'verbose': True}, scheduler_kwargs)

    def _(optim_args):
        return pyro.optim.PyroLRScheduler(
            lr_scheduler_cls, {'optimizer': optimizer_cls, 'optim_args': optim_args, **scheduler_kwargs},
            clip_args=clip_args
        )
    return _


class _call_forwarding(type):
    def __call__(cls, *args, **kwargs):
        return cls.__call__(*args, **kwargs)


class scheduled_optimizer_callback(metaclass=_call_forwarding):
    @staticmethod
    def _get_args(i, loss, locs):
        return ()

    @classmethod
    def __call__(cls, i, loss, locs):
        optim = locs['svi'].optim
        if isinstance(optim, pyro.optim.PyroLRScheduler):
            optim.step(*cls._get_args(i, loss, locs))


class scheduled_optimizer_callback_with_loss(scheduled_optimizer_callback):
    @staticmethod
    def _get_args(i, loss, locs):
        return loss,
