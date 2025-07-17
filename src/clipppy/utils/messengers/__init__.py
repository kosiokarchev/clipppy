from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from typing import Callable, Collection, Mapping, ContextManager, Iterable

import pyro.distributions
import torch
from more_itertools import last
from pyro.distributions import Uniform, ExpandedDistribution
from pyro.infer.autoguide.initialization import InitMessenger
from pyro.poutine import NonlocalExit
from pyro.poutine.escape_messenger import EscapeMessenger
from pyro.poutine.messenger import Messenger
from torch import Tensor
from torch.distributions import Independent
from torch.distributions.transforms import CumulativeDistributionTransform

from ..pyro import is_stochastic_site, make_deterministic
from ..typing import _Site
from ...distributions.wrapper import unwrap_only


@dataclass
class MultiContext(ExitStack):
    ctxs: Iterable[ContextManager]

    def __post_init__(self):
        super().__init__()

    def __enter__(self):
        for c in self.ctxs:
            self.enter_context(c)
        return super().__enter__()


class DeltaConditioningMessenger(Messenger):
    def __init__(self, data: Mapping[str, Tensor]):
        self.data = data

    def _pyro_sample(self, msg: _Site):
        if (name := msg['name']) in self.data:
            dta = self.data[name].expand(msg['fn'].event_shape)
            msg['fn'] = pyro.distributions.Delta(dta, event_dim=dta.ndim).expand(msg['fn'].batch_shape)


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

    def as_valuedict(self) -> dict[str, Tensor]:
        return {key: site['value'] for key, site in self.items() if 'value' in site}


class UnitCubePrior(CollectSitesMessenger):
    @staticmethod
    def _ucp_name(name):
        return f'_ucp_{name}'

    def _pyro_sample(self, msg: _Site):
        if msg['name'] in self.names:
            fn = msg['fn']
            make_deterministic(
                msg,
                CumulativeDistributionTransform(
                    last(unwrap_only(fn, (Independent, ExpandedDistribution)))
                ).inv(pyro.sample(
                    self._ucp_name(msg['name']),
                    Uniform(0, 1).expand(fn.shape()).to_event(fn.event_dim)
                )),
                fn.event_dim
            )


class SimpleReplayMessenger(Messenger):
    def __init__(self, values: Mapping[str, Tensor]):
        self.values = values

    def _pyro_sample(self, msg: _Site):
        if (name := msg['name']) in self.values:
            msg['value'] = self.values[name]
            msg['done'] = True


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
DetachAllMessenger = partial(ModifyValueMessenger, site_names=(), func=lambda x: x, func_other=Tensor.detach)


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


def apply_messenger(msgr):
    def _(func):
        def __(*args, **kwargs):
            with msgr:
                return func(*args, **kwargs)
        return __
    return _
