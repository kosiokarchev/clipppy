import typing as tp
from functools import partial

import torch
from pyro.infer.autoguide.initialization import InitMessenger
from pyro.poutine.condition_messenger import ConditionMessenger
from pyro.poutine.messenger import _bound_partial, _context_wrap, Messenger

from . import to_tensor
from .typing import _Site


__all__ = ('NoGradMessenger', 'init_fn', 'init_msgr', 'no_grad_msgr', 'depoutine')


class NoGradMessenger(Messenger):
    @staticmethod
    def _pyro_post_param(msg: _Site):
        msg['value'] = msg['value'].detach()


def init_fn(site: _Site) -> torch.Tensor:
    # sys.version_info >= (3, 8)
    # return init if (init := site['infer'].get('init', None)) is not None else site['fn']()
    init = site['infer'].get('init', None)
    return (to_tensor(init) if init is not None
            else site['fn']())


init_msgr = InitMessenger(init_fn)
no_grad_msgr = NoGradMessenger()


def depoutine(obj: tp.Union[_bound_partial, tp.Any], msgr_type: tp.Type[Messenger] = ConditionMessenger):
    # noinspection PyTypeHints
    obj.func: partial
    return obj.func.args[1] if (
        isinstance(obj, _bound_partial)
        and isinstance(obj.func, partial) and obj.func.func is _context_wrap
        and isinstance(obj.func.args[0], msgr_type)
    ) else obj
