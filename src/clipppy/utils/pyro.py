from __future__ import annotations

from abc import ABCMeta
from functools import partial, wraps
from typing import Any, Type, Union, Optional

import pyro
from pyro.distributions import Delta
from pyro.nn import PyroModule, PyroSample
from pyro.poutine.condition_messenger import ConditionMessenger
from pyro.poutine.messenger import _bound_partial, _context_wrap, Messenger
from pyro.poutine.runtime import _PYRO_STACK, am_i_wrapped
from pyro.poutine.trace_messenger import TraceMessenger
from torch import Tensor

from .typing import _Site


class AbstractPyroModuleMeta(type(PyroModule), ABCMeta):
    """"""
    def __getattr__(self, item):
        if item.startswith('_pyro_prior_'):
            return getattr(self, item.lstrip('_pyro_prior_')).prior
        return super().__getattr__(item)


class Contextful(property):
    def __set_name__(self, owner, name):
        self.name = name

    def _fget(self, instance: PyroModule):
        return self.fget(instance)

    def __get__(self, instance: Optional[PyroModule], owner: type[PyroModule] = None):
        if instance is None:
            return self
        name = instance._pyro_get_fullname(self.name)
        if (ret := instance._pyro_context.get(name)) is None:
            instance._pyro_context.set(name, ret := self._fget(instance))
        return ret


class PyroDeterministic(Contextful):
    def __init__(self, fget, event_dim=None):
        super().__init__(fget)
        self.event_dim = event_dim

    def _fget(self, instance: PyroModule):
        return pyro.deterministic(instance._pyro_get_fullname(self.name), super()._fget(instance), self.event_dim)



def depoutine(obj: Union[_bound_partial, Any], msgr_type: Type[Messenger] = ConditionMessenger):
    # noinspection PyTypeHints
    obj.func: partial
    return obj.func.args[1] if (
        isinstance(obj, _bound_partial)
        and isinstance(obj.func, partial) and obj.func.func is _context_wrap
        and isinstance(obj.func.args[0], msgr_type)
    ) else obj


def is_stochastic_site(site: _Site):
    return site['type'] == 'sample' and not (site['is_observed'] or site['infer'].get('_deterministic', False))


def make_deterministic(site: _Site, value: Tensor, event_dim: int = None):
    site['value'] = value
    site['fn'] = Delta(site['value'], event_dim=site['fn'].event_dim if event_dim is None else event_dim).mask(False)
    site['infer']['_deterministic'] = True


@wraps(pyro.sample)
def smart_sample(name, *args, **kwargs):
    if am_i_wrapped():
        for msgr in _PYRO_STACK:
            if isinstance(msgr, TraceMessenger) and name in msgr.trace.nodes:
                return msgr.trace.nodes[name]['value']
    return pyro.sample(name, *args, **kwargs)
