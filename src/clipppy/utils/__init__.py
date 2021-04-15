import enum
import re
import sys
import typing as tp
from functools import reduce
from itertools import chain, repeat
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Type, Union

import torch
from more_itertools import always_iterable, always_reversible, collapse, last, lstrip

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


def enumlstrip(iterable, pred):
    """lstrip with a pred that takes (index, value) as arguments"""
    for y in lstrip(enumerate(iterable), lambda ix: pred(*ix)):
        yield y[1]  # return value from (index, value)


def tryme(func: Callable[..., _T], exc: Type[Exception] = Exception, default: _T = None) -> _T:
    try:
        return func()
    except exc:
        return default


def noop(*args, **kwargs): pass


class Sentinel(enum.Enum):
    sentinel, skip, call, no_call = (object() for _ in range(4))

    def __repr__(self):
        return f'{type(self).__name__}.{self.name}'


# TODO: Decide on to_tensor strategy in general!
def to_tensor(val):
    return torch.tensor(val) if not torch.is_tensor(val) else val


_allmatch = re.compile('.*')
_nomatch = re.compile('.^')
