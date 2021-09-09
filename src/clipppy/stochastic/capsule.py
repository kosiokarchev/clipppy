from __future__ import annotations

from _weakref import ReferenceType
from typing import final, Generic, Optional, Type, TYPE_CHECKING, Union

from .wrapper import _cls, _T, CallableWrapper


@final
class Capsule(Generic[_T]):
    _value: Union[ReferenceType[_T], _T]
    __slots__ = '_value', 'lifetime', 'remaining'

    def __init__(self, lifetime: int = 1):
        self.lifetime = lifetime
        self.remaining = 0

    @classmethod
    def init(cls, value, lifetime: int = 1):
        self = cls(lifetime)
        self.value = value
        return self

    @property
    def value(self) -> _T:
        ret = self._value() if isinstance(self._value, ReferenceType) else self._value
        self.remaining -= 1
        if self.remaining < 1 and not isinstance(self._value, ReferenceType):
            try:
                self._value = ReferenceType(self._value)
            except TypeError:
                pass
        return ret

    @value.setter
    def value(self, val: _T):
        self._value = val
        self.remaining = self.lifetime

    def __call__(self):
        return self.value

    def __repr__(self):
        return f'<Capsule: {getattr(self, "_value", None)!r}>'


class AllEncapsulator(CallableWrapper[_T]):
    if TYPE_CHECKING:
        def __new__(cls: Type[_cls], obj: _T, *args, **kwargs) -> Union[_cls, _T]: ...

    __slots__ = 'capsule', 'capsule_args', 'capsule_kwargs'

    def _init__(self, obj, capsule: Optional[Capsule], /,
                *capsule_args: Capsule, **capsule_kwargs: Capsule):
        super().__init__(obj)
        self.capsule = capsule
        self.capsule_args = capsule_args
        self.capsule_kwargs = capsule_kwargs

    def __call__(self, *args, **kwargs):
        value = super().__call__(*args, **kwargs)
        if self.capsule:
            self.capsule.value = value
        if self.capsule_args:
            for capsule, val in zip(self.capsule_args, value):
                capsule.value = val
        if self.capsule_kwargs:
            for key in self.capsule_kwargs.keys() & value.keys():
                self.capsule_kwargs[key].value = value[key]
        return value


class Encapsulator(AllEncapsulator[_T]):
    if TYPE_CHECKING:
        def __new__(cls: Type[_cls], obj: _T, *args, **kwargs) -> Union[_cls, _T]: ...

    def _init__(self, obj, /, *capsule_args: Capsule, **capsule_kwargs: Capsule):
        super()._init__(obj, None, *capsule_args, **capsule_kwargs)
