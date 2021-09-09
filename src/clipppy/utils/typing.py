from __future__ import annotations

import abc
import sys
import types
from collections.abc import Callable
from typing import get_args, get_origin, Iterable, NewType, Optional, Pattern, Protocol, runtime_checkable, Type, TypedDict, TypeVar, Union

import torch
from pyro import distributions as dist
from pyro.poutine.indep_messenger import CondIndepStackFrame


__all__ = '_T', '_KT', '_VT', '_Tout', '_Tin', '_Site', '_Model', '_Guide'


_T = TypeVar('_T')
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')
_Tin = TypeVar('_Tin')
_Tout = TypeVar('_Tout')


# if sys.version_info < (3, 8):
#     def TypedDict(name: str, fields: Mapping[str, Any], total: bool = True):
#         return NewType(name, MutableMapping[str, Any])
#     TypedDict = TypedDict
#
#
#     class Literal(type):
#         @classmethod
#         def __getitem__(mcs, item):
#             typ = type(f'{mcs.__name__}[{repr(item)[1:-1]}]', (mcs,), {'__origin__': mcs})
#             typ.__args__ = item
#             return typ
#     Literal = Literal
#
#     get_origin = lambda generic: getattr(generic, '__origin__', None)
#     get_args = lambda generic: getattr(generic, '__args__', ())


if sys.version_info < (3, 9):
    class GenericAlias(abc.ABC):
        def __instancecheck__(self, instance):
            return get_origin(instance) and get_args(instance)

    types.GenericAlias = GenericAlias


@runtime_checkable
class Descriptor(Protocol[_T, _VT]):
    def __get__(self, instance: Optional[_T], owner: Type[_T]) -> _VT: ...


@runtime_checkable
class GetSetDescriptor(Protocol[_T, _VT]):
    def __get__(self, instance: Optional[_T], owner: Type[_T]) -> _VT: ...
    def __set__(self, instance: Optional[_T], value: _VT): ...


_Regex = Union[str, Pattern]

_Site = TypedDict('_Site', {
    'name': str, 'fn': dist.TorchDistribution, 'mask': torch.Tensor,
    'value': torch.Tensor, 'type': str, 'infer': dict,
    'cond_indep_stack': Iterable[CondIndepStackFrame]
}, total=False)
_Model = NewType('_Model', Callable)
_Guide = NewType('_Guide', Callable)
