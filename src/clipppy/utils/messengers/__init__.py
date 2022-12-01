from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Collection

import torch
from pyro.infer.autoguide.initialization import InitMessenger
from pyro.poutine import NonlocalExit
from pyro.poutine.escape_messenger import EscapeMessenger
from pyro.poutine.messenger import Messenger
from torch import Tensor

from ..pyro import is_stochastic_site
from ..typing import _Site


class PostEscapeMessenger(EscapeMessenger):
    def _pyro_sample(self, msg: _Site):
        return None

    def _pyro_post_sample(self, msg: _Site):
        return super()._pyro_sample(msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True if (
            ret := super().__exit__(exc_type, exc_val, exc_tb)
        ) is None and exc_type is NonlocalExit else ret


class UpToMessenger(PostEscapeMessenger):
    def escape_fn(self, msg: _Site):
        self.seen.add(msg['name'])
        return not bool(self.names - self.seen)

    def __init__(self, *names: str):
        super().__init__(self.escape_fn)
        self.names = set(names)
        self.seen = set()


class CollectSitesMessenger(UpToMessenger, dict[str, _Site]):
    def escape_fn(self, msg: _Site):
        if msg['name'] in self.names:
            self[msg['name']] = msg
        return super().escape_fn(msg)


@dataclass
class ModifyValueMessenger(Messenger):
    site_names: Collection[str]
    func: Callable[[Tensor], Tensor]
    other_condition: Callable[[_Site], bool] = staticmethod(is_stochastic_site)
    func_other: Callable[[Tensor], Tensor] = staticmethod(lambda x: x)

    def _pyro_post_sample(self, msg: _Site):
        if msg['name'] in self.site_names:
            msg['value'] = self.func(msg['value'])
        elif self.other_condition(msg):
            msg['value'] = self.func_other(msg['value'])


RequiresGradMessenger = partial(ModifyValueMessenger, func=partial(Tensor.requires_grad_, requires_grad=True))
DetachMessenger = partial(ModifyValueMessenger, func=Tensor.detach)


class NoGradMessenger(Messenger):
    @staticmethod
    def _pyro_post_param(msg: _Site):
        msg['value'] = new = (val := msg['value']).detach()

        # hack because Pyro hack .unconstrained onto the pure Tensor......
        if hasattr(val, 'unconstrained'):
            new.unconstrained = val.unconstrained


def init_fn(site: _Site) -> torch.Tensor:
    # return init if (init := site['infer'].get('init', None)) is not None else site['fn']()
    return site['infer'].get('init', None)


init_msgr = InitMessenger(init_fn)
no_grad_msgr = NoGradMessenger()
