from __future__ import annotations

import types
from functools import update_wrapper
from itertools import starmap
from typing import cast, ClassVar, Generic, MutableMapping, Type, TypeVar, Union

from frozendict import frozendict
from more_itertools import consume


_T = TypeVar('_T')
_cls = TypeVar('_cls')


class FunctionWrapper:
    """def functions and lambdas in python are special..."""
    def __init__(self, func: types.FunctionType):
        self.func = func
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Wrapper(Generic[_T]):
    """Maps non-wrapped types to wrapped with specific Wrapped subtype"""
    _wrapped_registry: ClassVar[MutableMapping[Type[_T], Type[Wrapper[_T]]]]

    def __init_subclass__(cls, final=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not final:
            cls._wrapped_registry = {}

    def __class_getitem__(cls, item: Type):
        if item not in cls._wrapped_registry:
            cls._wrapped_registry[item] = cast(Type[cls], type(f'{cls.__name__}[{item.__name__}]', (cls, item), {
                '__call__': update_wrapper(
                    lambda self, *args, **kwargs: super(type(self), self).__call__(*args, **kwargs),
                    item.__call__)
            }, final=True))
        return cls._wrapped_registry[item]

    _no_object = object()

    def __new__(cls: Type[_cls], obj: _T = _no_object, namespace=frozendict(), *args, **kwargs) -> Union[_cls, _T]:
        if obj is cls._no_object:
            return object.__new__(cls)
        if isinstance(obj, types.FunctionType):
            obj = FunctionWrapper(obj)

        self: Union[Wrapper, _T] = cls[type(obj)]()
        self.__dict__ = vars(obj)
        self._wrapped_obj = obj
        return self

    # Nothing should follow __new__, or super().__init__'s should stop here
    __init__ = lambda *args, **kwargs: None

    __slots__ = '_wrapped_obj'

    def __getstate__(self):
        return {attr: getattr(self, attr) for t in type(self).__mro__
                for attr in getattr(t, '__slots__', ()) if hasattr(self, attr)}

    def __setstate__(self, state):
        consume(starmap(self.__setattr__, state.items()))

    def __reduce__(self):
        wrapper_class = type(self).__bases__[0]
        return wrapper_class.__new__, (wrapper_class, self._wrapped_obj,), self.__getstate__()
