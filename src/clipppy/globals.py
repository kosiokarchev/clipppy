import builtins
import inspect
import operator
import sys

import typing as tp
from functools import reduce
from itertools import chain, repeat

import torch
import pyro.distributions as dist
from more_itertools import lstrip, last, always_reversible, always_iterable
from pyro.infer.autoguide.initialization import InitMessenger
from pyro.poutine.indep_messenger import CondIndepStackFrame


# TYPES
# -----
_T = tp.TypeVar('_T')
_KT = tp.TypeVar('_KT')
_VT = tp.TypeVar('_VT')
_Tin = tp.TypeVar('_Tin')
_Tout = tp.TypeVar('_Tout')


if sys.version_info < (3, 8):
    def TypedDict(name: str, fields: tp.Dict[str, tp.Any], total: bool = True):
        return tp.NewType(name, tp.Dict[str, tp.Any])
    tp.TypedDict = TypedDict

    class Literal(type):
        @classmethod
        def __getitem__(mcs, item):
            typ = type(f'{mcs.__name__}[{repr(item)[1:-1]}]', (mcs,), {'__origin__': mcs})
            typ.__args__ = item
            return typ

    tp.Literal = Literal

    tp.get_origin = lambda generic: getattr(generic, '__origin__', None)
    tp.get_args = lambda generic: getattr(generic, '__args__', ())

_Site = tp.TypedDict('_Site', {
    'name': str, 'fn': dist.TorchDistribution, 'mask': torch.Tensor,
    'value': torch.Tensor, 'type': str, 'infer': dict,
    'cond_indep_stack': tp.Iterable[CondIndepStackFrame]
}, total=False)

_Model = tp.NewType('_Model', tp.Callable)
_Guide = tp.NewType('_Guide', tp.Callable)


# SIGNATURES
# ----------
def is_variadic(param: inspect.Parameter) -> bool:
    return param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)


# GENERAL UTILITIES
# -----------------
def itemsetter(value=None, *keys, **kwargs):
    def _itemsetter(obj: tp.MutableMapping):
        for key, val in chain(zip(keys, repeat(value)), kwargs.items()):
            obj.__setitem__(key, val)
        return obj
    return _itemsetter


def flatten(*args: tp.Union[tp.Iterable, tp.Any]):
    for item in args:
        yield from flatten(*item) if isinstance(item, tp.Iterable) and not isinstance(item, str) else (item,)


def compose(*funcs: tp.Callable[[_Tin], tp.Union[_Tin, _Tout]]) -> tp.Callable[[_Tin], _Tout]:
    return lambda arg: last(arg for arg in (arg,) for f in always_reversible(flatten(funcs)) for arg in (f(arg),))


def valueiter(arg: tp.Union[tp.Iterable, tp.Mapping, tp.Any]):
    return (isinstance(arg, tp.Mapping) and arg.values()
            or always_iterable(arg))


def dict_union(*args: tp.Mapping[_KT, _VT], **kwargs: _VT) -> tp.Dict[_KT, _VT]:
    return reduce(lambda a, b: dict(chain(a.items(), b.items()))
                  if sys.version_info < (3, 9) else operator.or_,
                  args + (kwargs,), {})


def enumlstrip(iterable, pred):
    """lstrip with a pred that takes (index, value) as arguments"""
    for y in lstrip(enumerate(iterable), lambda ix: pred(*ix)):
        yield y[1]  # return value from (index, value)


def noop(*args, **kwargs): pass


def tryme(func: tp.Callable[..., _T], exc: tp.Type[Exception]=Exception, default: _T = None) -> _T:
    try:
        return func()
    except exc:
        return default


# GLOBAL REGISTRY
# ---------------
def get_global(name: str, default=None, scope: dict = None) -> tp.Any:
    if scope is not None and name in scope:
        return scope[name]
    elif name in globals():
        return globals()[name]
    elif hasattr(builtins, name):
        return getattr(builtins, name)
    else:
        return default


def register_globals(**kwargs):
    vars(builtins).update(kwargs)


# SPECIFIC UTILITIES
# ------------------
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
