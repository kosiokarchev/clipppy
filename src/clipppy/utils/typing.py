from __future__ import annotations

import abc
import re
import sys
import types
from collections.abc import Callable
from typing import (
    get_args, get_origin, Iterable, NewType, Optional, overload, Pattern,
    Protocol, runtime_checkable, Type, TypedDict, TypeVar, Union)

import torch
from more_itertools import collapse
from pyro import distributions as dist
from pyro.poutine.indep_messenger import CondIndepStackFrame
from typing_extensions import TypeAlias


__all__ = (
    '_T', '_KT', '_VT', '_Tout', '_Tin',
    'AnyRegex',
    '_Distribution', '_Site', '_Model', '_Guide'
)


_T = TypeVar('_T')
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')
_Tin = TypeVar('_Tin')
_Tout = TypeVar('_Tout')


if sys.version_info < (3, 9):
    class GenericAlias(abc.ABC):
        def __instancecheck__(self, instance):
            return get_origin(instance) and get_args(instance)

    types.GenericAlias = GenericAlias


@runtime_checkable
class SupportsItems(Protocol[_KT, _VT]):
    def items(self) -> Iterable[tuple[_KT, _VT]]: ...



@runtime_checkable
class Descriptor(Protocol[_T, _VT]):
    def __get__(self, instance: Optional[_T], owner: Type[_T]) -> _VT: ...


@runtime_checkable
class GetSetDescriptor(Protocol[_T, _VT]):
    def __get__(self, instance: Optional[_T], owner: Type[_T]) -> _VT: ...
    def __set__(self, instance: Optional[_T], value: _VT): ...


_Pattern: TypeAlias = Union[str, Pattern]
_Iterable_etc_of_Pattern: TypeAlias = Iterable[Union[_Pattern, '_Iterable_etc_of_Pattern']]
_AnyRegexable: TypeAlias = Union['AnyRegex', _Iterable_etc_of_Pattern]


class AnyRegex:
    def __init__(self, *patterns: _Pattern):
        self.patterns = tuple(map(re.compile, patterns))

    @classmethod
    @overload
    def get(cls, arg: AnyRegex) -> AnyRegex: ...

    @classmethod
    @overload
    def get(cls, *args: Union[_Pattern, _Iterable_etc_of_Pattern]) -> AnyRegex: ...

    @classmethod
    def get(cls, *args) -> AnyRegex:
        return args[0] if isinstance(args[0], AnyRegex) else AnyRegex(*collapse(args))

    def match(self, value: str, *args, **kwargs):
        return any(p.match(value, *args, **kwargs) for p in self.patterns)


_Distribution: TypeAlias = dist.TorchDistribution
_Site = TypedDict('_Site', {
    'done': bool,
    'name': str, 'fn': _Distribution, 'mask': torch.Tensor,
    'value': torch.Tensor, 'type': str, 'infer': dict, 'is_observed': bool,
    'cond_indep_stack': Iterable[CondIndepStackFrame]
}, total=False)
_Model = NewType('_Model', Callable)
_Guide = NewType('_Guide', Callable)
