from __future__ import annotations

from contextlib import nullcontext
from itertools import chain
from typing import Any, Callable, Collection, Iterable, Literal, Mapping, Type, TYPE_CHECKING, TypeVar, Union

from frozendict import frozendict
from more_itertools import always_iterable
from pyro.contrib.autoname import scope
from pyro.distributions.torch_distribution import TorchDistributionMixin

from .capsule import AllEncapsulator, Capsule
from .sampler import AbstractSampler, NamedSampler, PseudoSampler, Sampler
from ..utils import expandkeys, Sentinel
from ..utils.typing import SupportsItems


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

    def _scoped_call(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with scope(prefix=self.stochastic_name) if self.stochastic_name else nullcontext():
            return self._scoped_call(*args, **kwargs)


# TODO: StochasticSpecs type annotations
_SpecKT = Union[str, Literal[Sentinel.merge], Collection[str]]
_SpecVVT = Union[AbstractSampler, TorchDistributionMixin, Any]
_SpecVT = Union[_SpecVVT, Capsule, SupportsItems[str, _SpecVVT], Callable[[], SupportsItems[str, _SpecVVT]]]


class StochasticSpecs:
    def __init__(self, specs: Union[SupportsItems[_SpecKT, _SpecVT], Iterable[_SpecKT, _SpecVT]] = (), /, **kwargs: _SpecVT):
        self.specs: Iterable[tuple[_SpecKT, _SpecVT]] = [(
            name,
            (spec.set_name(strname) if spec.name is None else spec) if isinstance(spec, NamedSampler)
            else Sampler(spec, name=strname) if isinstance(spec, TorchDistributionMixin)
            else PseudoSampler(spec) if callable(spec) and not isinstance(spec, AbstractSampler)
            else spec
        ) for name, spec in chain(
            specs.items() if isinstance(specs, SupportsItems) else specs,
            kwargs.items()
        ) for strname in [name if isinstance(name, str) else None]
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

    def _init__(self, obj, specs: StochasticSpecs = None,
                capsule: Capsule = None, capsule_args: Iterable[Capsule] = (),
                capsule_kwargs: Mapping[str, Capsule] = frozendict(),
                name: str = None):
        self.stochastic_specs = specs if isinstance(specs, StochasticSpecs) else StochasticSpecs(specs)
        super()._init__(obj, name=name, capsule=capsule, capsule_args=capsule_args, capsule_kwargs=capsule_kwargs)

    def _scoped_call(self, *args, **kwargs):
        return super()._scoped_call(*args, **kwargs, **{
            key: (val() if isinstance(val, AbstractSampler) else
                  Sampler(val, name=key)() if isinstance(val, TorchDistributionMixin)
                  else val)
            for key, val in always_iterable(self.stochastic_specs)
            if key not in kwargs
        })

    def __repr__(self):
        return (f'"{self.stochastic_name}": ' if self.stochastic_name else '') + super().__repr__()