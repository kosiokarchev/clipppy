from dataclasses import dataclass
from typing import cast, ClassVar, MutableMapping, Type, TypeVar, Union

from clipppy.stochastic import StochasticWrapper
from clipppy.utils import wrap_any


class Wrapper:
    @dataclass
    class WrapperArgGetter:
        idx: int

        def __get__(self, instance, owner):
            if not isinstance(instance, Wrapper):
                raise TypeError(f'Can only get wrapper args on {type(Wrapper)} instances.')
            return instance._wrapper_args[self.idx]

    _wrapper_args: tuple
    _wrapped_obj = WrapperArgGetter(0)

    _wrapped_registry: ClassVar[MutableMapping[Type, Type[StochasticWrapper]]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._wrapped_registry = {}

    def __class_getitem__(cls, item: Type):
        _class = cls._wrapped_registry.get(item, None)
        if _class is None:
            _class = cast(Type[cls], type(f'{cls.__name__}[{item.__name__}]', (cls, item), {}))
            cls._wrapped_registry[item] = _class
        return _class

    @classmethod
    def _wrap(cls: Type[_cls], obj: _T, *args) -> Union[_cls, _T]:
        _obj: Union[Wrapper, _T] = wrap_any(obj)
        _obj.__class__ = cls[type(_obj)]
        _obj._wrapper_args = (obj,) + args
        return _obj

    def __reduce__(self):
        return type(self).mro()[1]._wrap, self._wrapper_args  # TODO: un-slight-hack


_T = TypeVar('_T')
_cls = TypeVar('_cls')