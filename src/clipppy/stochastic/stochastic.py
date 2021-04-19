from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Collection, Iterable, Literal, Mapping, Type, TYPE_CHECKING, TypeVar, Union

from frozendict import frozendict
from more_itertools import always_iterable
from pyro.contrib.autoname import scope
from pyro.distributions.torch_distribution import TorchDistributionMixin

from .capsule import AllEncapsulator, Capsule
from .sampler import AbstractSampler, NamedSampler, PseudoSampler, Sampler
from ..utils import expandkeys, PseudoString, Sentinel


_T = TypeVar('_T')
_cls = TypeVar('_cls')


__all__ = 'Stochastic', 'StochasticSpecs'


class StochasticScope(AllEncapsulator[_T]):
    __slots__ = 'stochastic_name'

    if TYPE_CHECKING:
        def __new__(cls: Type[_cls], obj: _T, *args, **kwargs) -> Union[_cls, _T]: ...

    def _init__(self, obj, name: str = None,
                capsule: Capsule = None, capsule_args: Iterable[Capsule] = (),
                capsule_kwargs: Mapping[str, Capsule] = frozendict()):
        self.stochastic_name = name
        super().__init__(obj, capsule, *capsule_args, **capsule_kwargs)

    def __call__(self, *args, **kwargs):
        with scope(prefix=self.stochastic_name) if self.stochastic_name else nullcontext():
            return super().__call__(*args, **kwargs)


_SpecKT = Union[str, Literal[Sentinel.merge], Collection[str]]
_SpecVVT = Union[AbstractSampler, TorchDistributionMixin, Any]
_SpecVT = Union[_SpecVVT, Capsule, Mapping[str, _SpecVVT], Callable[[], Mapping[str, _SpecVVT]]]


class StochasticSpecs:
    def __init__(self, **kwargs: _SpecVT):
        self.specs: Iterable[tuple[_SpecKT, _SpecVT]] = [(
            name,
            spec.set_name(strname) if isinstance(spec, NamedSampler)
            else Sampler(spec, name=strname) if isinstance(spec, TorchDistributionMixin)
            else PseudoSampler(spec) if callable(spec) and not isinstance(spec, PseudoSampler)
            else spec
        ) for name, spec in kwargs.items()
            for name in [name.meta if isinstance(name, PseudoString) else name]
            for strname in [name if isinstance(name, str) else None]
        ]

    def items(self) -> Iterable[tuple[str, _SpecVVT]]:
        return iter(self)

    def __iter__(self):
        for key, val in self.specs:
            if isinstance(key, str):
                yield key, val
            else:
                if callable(val):
                    val = val()
                yield from val.items() if key is Sentinel.merge else expandkeys(val, key)


class Stochastic(StochasticScope[_T]):
    __slots__ = 'stochastic_specs'

    if TYPE_CHECKING:
        def __new__(cls: Type[_cls], obj: _T, *args, **kwargs) -> Union[_cls, _T]: ...

    def _init__(self, obj, specs: StochasticSpecs = None, name: str = None,
                capsule: Capsule = None, capsule_args: Iterable[Capsule] = (),
                capsule_kwargs: Mapping[str, Capsule] = frozendict()):
        self.stochastic_specs = specs
        super()._init__(obj, name, capsule, *capsule_args, **capsule_kwargs)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs, **{
            key: (val() if isinstance(val, AbstractSampler) else
                  Sampler(val)() if isinstance(val, TorchDistributionMixin)
                  else val)
            for key, val in always_iterable(self.stochastic_specs)
            if key not in kwargs
        })

    def __repr__(self):
        return (f'"{self.stochastic_name}": ' if self.stochastic_name else '') + super().__repr__()