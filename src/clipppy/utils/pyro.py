from __future__ import annotations

from abc import ABCMeta
from functools import partial, wraps
from typing import Any, Type, Union

import pyro
from pyro.nn import PyroModule
from pyro.poutine.condition_messenger import ConditionMessenger
from pyro.poutine.messenger import _bound_partial, _context_wrap, Messenger
from pyro.poutine.runtime import _PYRO_STACK, am_i_wrapped
from pyro.poutine.trace_messenger import TraceMessenger

from .typing import _Site


class AbstractPyroModuleMeta(type(PyroModule), ABCMeta):
    """"""
    def __getattr__(self, item):
        if item.startswith('_pyro_prior_'):
            return getattr(self, item.lstrip('_pyro_prior_')).prior
        return super().__getattr__(item)


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


@wraps(pyro.sample)
def smart_sample(name, *args, **kwargs):
    if am_i_wrapped():
        for msgr in _PYRO_STACK:
            if isinstance(msgr, TraceMessenger) and name in msgr.trace.nodes:
                return msgr.trace.nodes[name]['value']
    return pyro.sample(name, *args, **kwargs)
