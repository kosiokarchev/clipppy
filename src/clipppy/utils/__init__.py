import operator
import re
import sys
import typing as tp
from functools import reduce
from itertools import chain, repeat

import torch
from more_itertools import always_iterable, always_reversible, last, lstrip

from .typing import _KT, _T, _Tin, _Tout, _VT


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
    return isinstance(arg, tp.Mapping) and arg.values() or always_iterable(arg)


def dict_union(*args: tp.Mapping[_KT, _VT], **kwargs: _VT) -> tp.Dict[_KT, _VT]:
    return reduce((lambda a, b: dict(chain(a.items(), b.items())))
                  if sys.version_info < (3, 9) else operator.or_,
                  args + (kwargs,), {})


def enumlstrip(iterable, pred):
    """lstrip with a pred that takes (index, value) as arguments"""
    for y in lstrip(enumerate(iterable), lambda ix: pred(*ix)):
        yield y[1]  # return value from (index, value)


def tryme(func: tp.Callable[..., _T], exc: tp.Type[Exception]=Exception, default: _T = None) -> _T:
    try:
        return func()
    except exc:
        return default


def noop(*args, **kwargs): pass


# TODO: Decide on to_tensor strategy in general!
def to_tensor(val):
    return torch.tensor(val) if not torch.is_tensor(val) else val


_allmatch = re.compile('.*')
_nomatch = re.compile('.^')
