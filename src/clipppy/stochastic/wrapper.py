from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from functools import update_wrapper
from itertools import starmap
from types import FunctionType
from typing import cast, ClassVar, Generic, MutableMapping, Type, TypeVar, Union

from more_itertools import consume

from ..utils import copy_function


_T = TypeVar('_T')
_cls = TypeVar('_cls')


class FunctionWrapper:
    """def functions and lambdas in python are special..."""
    def __init__(self, func: FunctionType):
        self.func = func
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Wrapper(Generic[_T]):
    """Maps non-wrapped types to wrapped with specific Wrapped subtype"""
    _wrapped_registry: ClassVar[MutableMapping[Type[_T], Type[Wrapper[_T]]]]

    def __init_subclass__(cls, final=False, **kwargs):
        super().__init_subclass__(**kwargs)

        # TODO: Hack type-checker (PyCharm) better
        if hasattr(cls, '_init__'):
            cls.__init__ = cls._init__
            # del cls._init__

        if not final:
            cls._wrapped_registry = {}
            cls.__new__ = update_wrapper(copy_function(cls.__new__), cls.__init__)

    def __class_getitem__(cls, item: Type):
        if isinstance(item, TypeVar):
            return super().__class_getitem__(item)
        if item not in cls._wrapped_registry:
            cls._wrapped_registry[item] = cast(Type[cls], type(f'{cls.__name__}[{item.__name__}]', (cls, item), {}, final=True))
        return cls._wrapped_registry[item]

    def __new__(cls: Type[_cls], obj: _T, /, *args, **kwargs) -> Union[_cls, _T]:
        if isinstance(obj, FunctionType):
            obj = FunctionWrapper(obj)

        self: Union[Wrapper, _T] = object.__new__(cls[type(obj)])
        self.__dict__ = vars(obj)
        self._wrapped_obj = obj
        return self

    # super().__init__'s should stop here
    def __init__(self, *args, **kwargs): pass

    __slots__ = '_wrapped_obj', '_call__'

    def __getstate__(self):
        return {attr: getattr(self, attr) for t in type(self).__mro__
                for attr in getattr(t, '__slots__', ()) if hasattr(self, attr)}

    def __setstate__(self, state):
        consume(starmap(self.__setattr__, state.items()))

    def __reduce__(self):
        wrapper_class = type(self).__bases__[0]
        return wrapper_class.__new__, (wrapper_class, self._wrapped_obj,), self.__getstate__()


class CallableWrapper(Wrapper[_T], ABC):
    __slots__ = '_call__'

    def __getattribute__(self, item):
        if item == '__call__':
            return self._call__
        return super().__getattribute__(item)

    def __init__(self, obj, /, *args, **kwargs):
        self._call__ = update_wrapper(copy(type(self).__call__), type(obj).__call__)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
