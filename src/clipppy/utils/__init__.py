from __future__ import annotations

import enum
import re
from itertools import chain, repeat
from types import FunctionType
from typing import Any, Callable, Collection, Generic, Iterable, Literal, Mapping, MutableMapping, Type, Union

import torch
from more_itertools import always_iterable, always_reversible, collapse, last, lstrip, padded, spy

from .typing import _KT, _T, _T1, _T2, _Tin, _Tout, _VT, SupportsItems


def itemsetter(value=None, *keys, **kwargs):
    def _itemsetter(obj: MutableMapping):
        for key, val in chain(zip(keys, repeat(value)), kwargs.items()):
            obj.__setitem__(key, val)
        return obj
    return _itemsetter


def caller(obj):
    return obj()


def compose(*funcs: Callable[[_Tin], Union[_Tin, _Tout]]) -> Callable[[_Tin], _Tout]:
    return lambda arg: last(arg for arg in (arg,) for f in always_reversible(collapse(funcs)) for arg in (f(arg),))


def valueiter(arg: Union[Iterable[_T], Mapping[Any, _T], _T]) -> Iterable[_T]:
    return isinstance(arg, Mapping) and arg.values() or always_iterable(arg)


def filterkeys(f: Callable[[_KT], bool], m: Union[Mapping[_KT, _VT], Iterable[tuple[_KT, _VT]]]) -> Iterable[tuple[_KT, _VT]]:
    return (kv for kv in (m.items() if isinstance(m, Mapping) else m) if f(kv[0]))


def expandkeys(m: Union[SupportsItems[_KT, _VT], Iterable[tuple[_KT, _VT]], Iterable[_VT]], keys: Collection[_KT]):
    """Return specific keys from a mapping or iterable of key-value pairs, or zip them with a value iterator."""
    if isinstance(m, Mapping):
        m = m.items()
    (f,), m = spy(m)
    return (el for el in m if el[0] in keys) if isinstance(f, tuple) and len(f) == 2 else ((k, next(m)) for k in keys)


def enumlstrip(iterable, pred):
    """lstrip with a pred that takes (index, value) as arguments"""
    for y in lstrip(enumerate(iterable), lambda ix: pred(*ix)):
        yield y[1]  # return value from (index, value)


def copy_function(f: FunctionType, name=None):
    return FunctionType(f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__)


def zip_asymmetric(arg1: Iterable[_T1], arg2: Iterable[_T2], err: Exception) -> Iterable[tuple[_T1, _T2]]:
    sentinel = object()
    for a1, a2 in zip(arg1, padded(arg2, sentinel)):
        if a2 is sentinel:
            raise err
        yield a1, a2



def tryme(func: Callable[..., _T], exc: Type[Exception] = Exception, default: _T = None) -> _T:
    try:
        return func()
    except exc:
        return default


# noinspection PyUnusedLocal
def noop(*args, **kwargs): pass


class Sentinel(enum.Enum):
    sentinel, skip, empty, call, no_call, pos, mergepos, merge = (object() for _ in range(8))

    def __repr__(self):
        return f'{type(self).__name__}.{self.name}'


def merge_if_not_skip(a: Mapping[_KT, _VT], b: Mapping[_KT, Union[_VT, Literal[Sentinel.skip]]]) -> Mapping[_KT, _VT]:
    return {**a, **dict(filter(lambda keyval: keyval[1] is not Sentinel.skip, b.items()))}


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


def torch_get_default_device():
    return torch._C._get_default_device()


_allmatch = re.compile('.*')
_nomatch = re.compile('.^')


def _detensorify(t):
    return t.cpu() if torch.is_tensor(t) else t


def call_nontensor(func, *args, **kwargs):
    all_args = tuple(chain(args, kwargs.values()))
    if any((torch.is_tensor(arg) and arg.requires_grad) for arg in all_args):
        raise NotImplementedError
    extensor = next(filter(torch.is_tensor, all_args))
    return torch.as_tensor(
        func(*map(_detensorify, args), **dict(zip(kwargs.keys(), map(_detensorify, kwargs.values())))),
        dtype=extensor.dtype, device=extensor.device
    )
