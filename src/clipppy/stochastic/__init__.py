from __future__ import annotations

from _weakref import ReferenceType
from contextlib import nullcontext
from typing import (Any, Callable, final, Generic, Iterable, Mapping, Optional, Type, TYPE_CHECKING, Union)

from frozendict import frozendict
from pyro.contrib.autoname import scope
from pyro.distributions.torch_distribution import TorchDistributionMixin

from .infinite import *
from .sampler import AbstractSampler, Sampler
from .wrapper import _cls, _T, CallableWrapper, Wrapper


__all__ = 'stochastic',


@final
class Capsule(Generic[_T]):
    _value: Union[ReferenceType[_T], _T]
    __slots__ = '_value', 'lifetime', 'remaining'

    def __init__(self, lifetime: int = 1):
        self.lifetime = lifetime
        self.remaining = 0

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

    def __repr__(self):
        return f'<Capsule: {getattr(self, "_value", None)!r}>'


class Encapsulator(CallableWrapper[_T]):
    __slots__ = 'capsule', 'capsule_args', 'capsule_kwargs'

    def __init__(self, obj, /, *capsule_args: Capsule, **capsule_kwargs: Capsule):
        super().__init__(obj)
        self.capsule: Optional[Capsule] = None
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


class AllEncapsulator(Encapsulator[_T]):
    def __init__(self, obj, capsule: Capsule, /, *capsule_args: Capsule, **capsule_kwargs: Capsule):
        super().__init__(obj, *capsule_args, **capsule_kwargs)
        self.capsule = capsule


_SpecT = Union[Sampler, Capsule, TorchDistributionMixin, Any]


class Stochastic(AllEncapsulator[_T]):
    __slots__ = 'stochastic_specs', 'stochastic_name'

    if TYPE_CHECKING:
        def __new__(cls: Type[_cls], obj: _T, *args, **kwargs): ...
    else:
        def _init__(self: _cls, obj: _T, specs: Mapping[str, _SpecT] = frozendict(), name: str = None,
                    capsule: Capsule = None, capsule_args: Iterable[Capsule] = (),
                    capsule_kwargs: Mapping[str, Capsule] = frozendict()):
            self.stochastic_specs = specs
            self.stochastic_name = name
            super().__init__(obj, capsule, *capsule_args, **capsule_kwargs)

        __init__ = _init__

    def __call__(self, *args, **kwargs):
        with scope(prefix=self.stochastic_name) if self.stochastic_name else nullcontext():
            return super().__call__(*args, **kwargs, **{
                name: (
                    spec() if isinstance(spec, AbstractSampler) else
                    spec.value if isinstance(spec, Capsule) else
                    Sampler(spec, name=name) if isinstance(spec, TorchDistributionMixin)
                    else spec)
                for name, spec in self.stochastic_specs.items()
                if name not in kwargs
            })

    def __repr__(self):
        return (f'"{self.stochastic_name}": ' if self.stochastic_name else '') + super().__repr__()


s = Stochastic(lambda: None)


def stochastic(obj: Callable, specs: Mapping[str, Union[_SpecT, TorchDistributionMixin]],
               name: str = None):
    r"""
    Make a StochasticWrapper from a dict of specs that can be:
       - full-blown `AbstractSampler`\ s
       - `TorchDistributionMixin`\ s that will be wrapped in `Sampler`\ s
       - any other value that will be passed as-is
    If you really want to pass a distribution to the wrapper as-is,
    use `StochasticWrapper._wrap`.
    """
    return Stochastic(obj=obj, specs={
        name: (spec.set_name(name) if isinstance(spec, AbstractSampler)
               else Sampler(spec, name=name) if isinstance(spec, TorchDistributionMixin)
               else spec)
        for name, spec in specs.items()
    }, name=name)
