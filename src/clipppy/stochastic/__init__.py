from __future__ import annotations

from contextlib import nullcontext
from typing import Any, cast, Mapping, MutableMapping, Type, TypeVar, Union

from pyro import distributions as dist
from pyro.contrib.autoname import scope

from .sampler import AbstractSampler, Sampler
from ..utils import wrap_any


__all__ = 'stochastic',

_T = TypeVar('_T')
_cls = TypeVar('_cls')


class StochasticWrapper:
    _wrapped_registry: MutableMapping[Type, Type[StochasticWrapper]] = {}

    _stochastic_obj: Any
    stochastic_specs: Mapping[str, Union[Sampler, Any]]
    stochastic_name: str

    @property
    def _stochastic_args(self):
        return self._stochastic_obj, self.stochastic_specs, self.stochastic_name

    @_stochastic_args.setter
    def _stochastic_args(self, value):
        self._stochastic_obj, self.stochastic_specs, self.stochastic_name = value

    def __call__(self, *args, **kwargs):
        with scope(prefix=self.stochastic_name) if self.stochastic_name else nullcontext():
            return super().__call__(*args, **kwargs, **{
                name: (spec() if isinstance(spec, AbstractSampler) else spec)
                for name, spec in self.stochastic_specs.items()
                if name not in kwargs
            })

    def __repr__(self):
        return (f'"{self.stochastic_name}": ' if self.stochastic_name else '') + super().__repr__()

    def __reduce__(self):
        return type(self).__mro__[1].stochastic_wrap, self._stochastic_args

    def __class_getitem__(cls, item: Type):
        _class = cls._wrapped_registry.get(item, None)
        if _class is None:
            _class = cast(Type[cls], type(f'{cls.__name__}[{item.__name__}]', (cls, item), {}))
            cls._wrapped_registry[item] = _class
        return _class

    @classmethod
    def stochastic_wrap(cls: Type[_cls], obj: _T, stochastic_specs: Mapping[str, Union[Sampler, Any]], name=None) -> Union[_cls, _T]:
        """Make a StochasticWrapper from an object, with the given specs and name."""
        _obj: Union[_cls, _T] = wrap_any(obj)
        _obj.__class__ = cls[type(_obj)]
        _obj._stochastic_args = obj, stochastic_specs, name
        return _obj


def stochastic(obj, specs: Mapping[str, Union[Sampler, dist.torch_distribution.TorchDistributionMixin, Any]], name=None):
    """
    Make a StochasticWrapper from a dict of specs that can be:
       - full-blown Sampler's
       - TorchDistributionMixin's that will be wrapped in Sampler's
       - any other value that will be passed as-is
    If you really want to pass a distribution to the wrapper as-is,
    use :any:`_stochastic`.
    """
    return StochasticWrapper.stochastic_wrap(obj=obj, stochastic_specs={
        name: (spec.set_name(name) if isinstance(spec, AbstractSampler)
               else Sampler(spec, name=name) if isinstance(spec, dist.torch_distribution.TorchDistributionMixin)
               else spec)
        for name, spec in specs.items()
    }, name=name)
