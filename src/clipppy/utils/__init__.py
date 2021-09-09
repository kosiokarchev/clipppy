from __future__ import annotations

import enum
import re
from itertools import chain, repeat
from types import FunctionType
from typing import Any, Callable, Collection, Generic, Iterable, Mapping, MutableMapping, Type, Union

import torch
from more_itertools import always_iterable, always_reversible, collapse, last, lstrip, spy

from .typing import _KT, _T, _Tin, _Tout, _VT


def itemsetter(value=None, *keys, **kwargs):
    def _itemsetter(obj: MutableMapping):
        for key, val in chain(zip(keys, repeat(value)), kwargs.items()):
            obj.__setitem__(key, val)
        return obj
    return _itemsetter


def compose(*funcs: Callable[[_Tin], Union[_Tin, _Tout]]) -> Callable[[_Tin], _Tout]:
    return lambda arg: last(arg for arg in (arg,) for f in always_reversible(collapse(funcs)) for arg in (f(arg),))


def valueiter(arg: Union[Iterable[_T], Mapping[Any, _T], _T]) -> Iterable[_T]:
    return isinstance(arg, Mapping) and arg.values() or always_iterable(arg)


def filterkeys(f: Callable[[_KT], bool], m: Union[Mapping[_KT, _VT], Iterable[tuple[_KT, _VT]]]) -> Iterable[tuple[_KT, _VT]]:
    return (kv for kv in (m.items() if isinstance(m, Mapping) else m) if f(kv[0]))


def expandkeys(m: Union[Mapping[_KT, _VT], Iterable[tuple[_KT, _VT]], Iterable[_VT]], keys: Collection[_KT]):
    """Return specific keys from a mapping or iterable of key-value pairs, or zip them with a value iterator."""
    if isinstance(m, Mapping):
        m = m.items()
    (f,), m = spy(m)
    return ((k, next(m)) for k in keys) if len(f) == 1 else (el for el in m if el[0] in keys)


def enumlstrip(iterable, pred):
    """lstrip with a pred that takes (index, value) as arguments"""
    for y in lstrip(enumerate(iterable), lambda ix: pred(*ix)):
        yield y[1]  # return value from (index, value)


def copy_function(f: FunctionType, name=None):
    return FunctionType(f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__)



def tryme(func: Callable[..., _T], exc: Type[Exception] = Exception, default: _T = None) -> _T:
    try:
        return func()
    except exc:
        return default


# noinspection PyUnusedLocal
def noop(*args, **kwargs): pass


class Sentinel(enum.Enum):
    sentinel, skip, call, no_call, pos, mergepos, merge = (object() for _ in range(7))

    def __repr__(self):
        return f'{type(self).__name__}.{self.name}'


class PseudoString(str, Generic[_T]):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    # noinspection PyUnusedLocal
    def __init__(self, meta: _T, *args, **kwargs):
        self.meta = meta

    @classmethod
    def init(cls, *args: _Tin) -> PseudoString[tuple[_Tin]]:
        return cls(args)

    def __hash__(self):
        return object.__hash__(self)


# TODO: Decide on to_tensor strategy in general!
def to_tensor(val):
    return torch.tensor(val) if not torch.is_tensor(val) else val


_allmatch = re.compile('.*')
_nomatch = re.compile('.^')
