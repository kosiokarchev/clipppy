from __future__ import annotations

import enum
import re
from itertools import chain
from types import FunctionType
from typing import Callable, Collection, Generic, Iterable, Literal, Mapping, Type, Union
from warnings import filterwarnings, catch_warnings

import torch
from more_itertools import padded, spy

import phytorchx
from .typing import _KT, _T, _T1, _T2, _Tin, _Tout, _VT, SupportsItems


def caller(obj):
    return obj()


def expandkeys(m: Union[SupportsItems[_KT, _VT], Iterable[tuple[_KT, _VT]], Iterable[_VT]], keys: Collection[_KT]):
    """Return specific keys from a mapping or iterable of key-value pairs, or zip them with a value iterator."""
    if isinstance(m, Mapping):
        m = m.items()
    (f,), m = spy(m)
    return (el for el in m if el[0] in keys) if isinstance(f, tuple) and len(f) == 2 else ((k, next(m)) for k in keys)


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

    # TODO: >=3.11: enum.pickle_by_enum_name
    def __reduce_ex__(self, proto):
        return getattr, (self.__class__, self._name_)

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
    return torch.tensor(val, dtype=torch.get_default_dtype(), device=phytorchx.get_default_device()) if not torch.is_tensor(val) else val


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


def log_prob_to_cred(lp: Tensor, ndim: int = None):
    with catch_warnings():
        filterwarnings(action='ignore', message='Named tensors')

        start_dim = lp.ndim - (ndim or lp.ndim)
        flat = lp.rename(None).flatten(start_dim)
        argsort = flat.argsort(-1, descending=True)
        return (
            torch.empty_like(flat, memory_format=torch.contiguous_format)
            .scatter_(
                -1, argsort,
                (flat.take_along_dim(argsort, -1).logcumsumexp(-1) - flat.logsumexp(-1, keepdim=True))
            ).unflatten(-1, lp.shape[start_dim:]).rename_(*lp.names)
        )
