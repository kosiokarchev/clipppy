from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, ClassVar, Generic, Mapping, Type, Union

from frozendict import frozendict
from pyro.contrib.autoname import scope
from pyro.distributions import TorchDistributionMixin

from . import _T, Wrapper
from .sampler import AbstractSampler, Sampler
from .wrapper import _cls, _T, Wrapper


__all__ = 'stochastic',


class Capsule(Generic[_T]):
    always_include_self: ClassVar = False
    value: _T

    def __init__(self, *arg_capsules: Capsule, **kwarg_capsules: Capsule):
        self.arg_capsules = arg_capsules
        self.kwarg_capsules = kwarg_capsules

    def fill(self, value):
        if self.always_include_self or not (self.arg_capsules or self.kwarg_capsules):
            self.value = value
        for capsule, val in zip(self.arg_capsules, value):
            capsule.value = val
        for key in self.kwarg_capsules.keys() & value.keys():
            self.kwarg_capsules[key].value = value[key]


class ACapsule(Capsule):
    always_include_self = True


_SpecT = Union[Sampler, Any]

# _ExpandSentinel = NewType('_ExpandSentinel', object)
# expand_sentinel: Final = _ExpandSentinel(object())
#
# _SpecKT: TypeAlias = str
# _SpecVT = Union[Sampler, TorchDistributionMixin, Callable[[], 'SpecT']]
# _SpecT = Mapping[_SpecKT, _SpecVT]
#
# def expanded_items(m: Mapping[Union[Any, _ExpandSentinel], Union[Any, Callable[[], Mapping[]]]]):
#     for key, value in m.items():
#         if key is expand_sentinel:
#             yield
#         yield (key, value)


class StochasticWrapper(Wrapper):
    stochastic_specs: Mapping[str, _SpecT] = Wrapper.WrapperArgGetter(1)
    stochastic_capsule: Capsule = Wrapper.WrapperArgGetter(2)
    stochastic_name: str = Wrapper.WrapperArgGetter(3)

    @classmethod
    def _wrap(cls: Type[_cls], obj: _T, specs: Mapping[str, _SpecT] = frozendict(), capsule: Capsule = None, name=None):
        return super()._wrap(obj, specs, capsule, name)

    expand: ClassVar = object()

    def __call__(self, *args, **kwargs):
        with scope(prefix=self.stochastic_name) if self.stochastic_name else nullcontext():
            return super().__call__(*args, **kwargs, **{
                name: (spec() if isinstance(spec, AbstractSampler) else spec)
                for name, spec in self.stochastic_specs.items()
                if name not in kwargs
            })

    def __repr__(self):
        return (f'"{self.stochastic_name}": ' if self.stochastic_name else '') + super().__repr__()


def stochastic(obj: Callable, specs: Mapping[str, Union[_SpecT, TorchDistributionMixin]],
               capsule: Capsule = None, name: str = None):
    r"""
    Make a StochasticWrapper from a dict of specs that can be:
       - full-blown `AbstractSampler`\ s
       - `TorchDistributionMixin`\ s that will be wrapped in `Sampler`\ s
       - any other value that will be passed as-is
    If you really want to pass a distribution to the wrapper as-is,
    use `StochasticWrapper._wrap`.
    """
    return StochasticWrapper._wrap(obj=obj, specs={
        name: (spec.set_name(name) if isinstance(spec, AbstractSampler)
               else Sampler(spec, name=name) if isinstance(spec, TorchDistributionMixin)
               else spec)
        for name, spec in specs.items()
    }, capsule=capsule, name=name)
