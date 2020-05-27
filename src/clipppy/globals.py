import builtins
import sys

import typing as tp
from itertools import chain, repeat

import torch
import pyro.distributions as dist
from more_itertools import lstrip
from pyro.infer.autoguide.initialization import InitMessenger
from pyro.poutine.indep_messenger import CondIndepStackFrame


def get_global(name: str, default=None, scope: dict = None) -> tp.Any:
    if sys.version_info >= (3, 9):
        return (globals() | (scope if scope is not None else {})).get(name, getattr(builtins, name, default))
    else:
        scope = scope.copy() if scope is not None else {}
        scope.update(globals())
        return scope.get(name, getattr(builtins, name, default))


def register_globals(**kwargs):
    for key, value in kwargs.items():
        setattr(builtins, key, value)


def itemsetter(value=None, *keys, **kwargs):
    def _itemsetter(obj: tp.MutableMapping):
        for key, val in chain(zip(keys, repeat(value)), kwargs.items()):
            obj.__setitem__(key, val)
        return obj
    return _itemsetter


def enumlstrip(iterable, pred):
    """lstrip with a pred that takes (index, value) as arguments"""
    for y in lstrip(enumerate(iterable), lambda ix: pred(*ix)):
        yield y[1]  # return value from (index, value)


if sys.version_info >= (3, 8):
    _Site = tp.TypedDict('_Site', {
        'name': str, 'fn': dist.TorchDistribution, 'mask': torch.Tensor,
        'value': torch.Tensor, 'type': str, 'infer': dict,
        'cond_indep_stack': tp.Iterable[CondIndepStackFrame]
    }, total=False)
else:
    # pass
    _Site = tp.Dict


def to_tensor(val):
    return torch.tensor(val) if not torch.is_tensor(val) else val


# TODO: Decide on to_tensor strategy in general!
def init_fn(site: _Site) -> torch.Tensor:
    # sys.version_info >= (3, 8)
    # return init if (init := site['infer'].get('init', None)) is not None else site['fn']()
    init = site['infer'].get('init', None)
    return (to_tensor(init) if init is not None
            else site['fn']())


init_msgr = InitMessenger(init_fn)
