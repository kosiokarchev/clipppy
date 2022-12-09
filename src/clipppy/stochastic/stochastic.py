from __future__ import annotations

from contextlib import nullcontext
from itertools import chain
from operator import itemgetter
from typing import Any, Callable, Collection, Iterable, Literal, Mapping, Tuple, Type, TYPE_CHECKING, TypeVar, Union
from warnings import warn

from frozendict import frozendict
from pyro.contrib.autoname import scope
from pyro.distributions.torch_distribution import TorchDistributionMixin
from typing_extensions import TypeAlias

from .capsule import AllEncapsulator, Capsule
from .sampler import AbstractSampler, NamedSampler, PseudoSampler, Sampler
from ..utils import expandkeys, Sentinel
from ..utils.typing import SupportsItems


_T = TypeVar('_T')
_cls = TypeVar('_cls')


__all__ = 'Stochastic', 'StochasticSpecs', 'StochasticScope'


class StochasticScope(AllEncapsulator[_T]):
    """A wrapper that introduces a `pyro.contrib.autoname.scope`."""

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


_SpecT: TypeAlias = Union[AbstractSampler, TorchDistributionMixin, Any]
_eSpecT: TypeAlias = Union[SupportsItems[str, _SpecT], Iterable[Tuple[str, _SpecT]], Iterable[_SpecT]]
_SpecKT: TypeAlias = Union[str, Literal[Sentinel.merge], Collection[str]]
_SpecVT: TypeAlias = Union[_SpecT, _eSpecT, Callable[[], _eSpecT]]


class StochasticSpecs:
    r"""A "mapping" of parameter names to "specifications" used in `Stochastic`.

    When a `StochasticSpecs` is created, it is provided with a mapping from
    names to "specifiations". Each "specification" can be one of
        - an instance of `AbstractSampler`. In that case, its name is set to
          the corresponding key if the specification is a `NamedSampler`;
        - a `~pyro.distributions.torch_distribution.TorchDistributionMixin`.
          It is wrapped in a `Sampler`, which is also named;
        - a `callable` (that is not an `AbstractSampler`). It is wrapped in a
          `PseudoSampler`.
        - any other value is not modified.

    To support merging specifications from multiple mappings, a key can also be
        - `Sentinel.merge`: the specification is then assumed to be a mapping
          (in fact, `SupportsItems`) and its items merged in;
        - or a collection of strings that are either extracted from the
          specification if it is a mapping, or are zipped against it if it is
          another iterable of values.
    In both cases, if the specification is callable, it is **called**, and the
    resulting object is used for merging.

    Notes
    -----
    `StochasticSpecs` is primarily intended to be automatically created by
    |Clipppy|'s YAML perser, which contains a hook for transforming the YAML
    merge magic key (`<<`) into the `Sentinel.merge` token. Additionally, the
    logic is such as to allow easy specification of `Capsule`\ s in YAML: their
    `~Capsule.value`\ s are then automatically extracted by the rules outlined
    above.
    """
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
        ) for strname in [name if isinstance(name, str) else None]]

    # TODO: StochasticSpecs Mapping interface
    # TODO: StochasticSpecs -> dict conversion convention

    def __getitem__(self, item):
        warn(f'Getting items from {type(self).__name__} by name is frowned upon'
             ' and only supports explicitly named specs at the first level that'
             ' at most come from mappings (no dynamic generation).',
             RuntimeWarning)

        for key, val in self.specs:
            if key == item:
                return val
        raise KeyError(item)

    def __setitem__(self, key, value):
        warn(f'Setting items on {type(self).__name__} by name is frowned upon'
             ' and only supports explicitly named specs at the first level.',
             RuntimeWarning)

        for i, _key in enumerate(map(itemgetter(0), self.specs)):
            if key == _key:
                self.specs[i] = (key, value)
                return
        self.specs.append((key, value))

    # def keys(self):
    #     warn(f'Iterating keys from {type(self).__name__} is frowned upon'
    #          ' and returns the raw keys!', RuntimeWarning)
    #     return map(itemgetter(0), self.specs)

    def values(self):
        warn(f'Iterating values from {type(self).__name__} is frowned upon'
             ' and returns the raw specs!', RuntimeWarning)
        return map(itemgetter(1), self.specs)

    def items(self) -> Iterable[tuple[str, _SpecT]]:
        r"""Iterate the key-"specification" pairs via `iter`\ ``(self)``.

        Provided for compatibiility with other mappings (i.e. with the
        `SupportsItems` protocol)."""
        return iter(self)

    def __iter__(self):
        """Iterate the key-"specification" pairs."""
        for key, val in self.specs:
            if isinstance(key, str):
                yield key, val
            else:
                if callable(val):
                    val = val()
                yield from val.items() if key is Sentinel.merge else expandkeys(val, key)


class Stochastic(StochasticScope[_T]):
    """A wrapper that generates (possibly stochastic) parameters for each invokation.

    A `Stochastic` is similar to `~functools.partial` in that it supplies some
    or all parameters to the underlying callable. The main differences are two:
    first, `Stochastic` supports **only keyword** parameters; and secondly,
    whereas the parameters that `~functools.partial` passes are invariant,
    `Stochastic` generates them anew for each invokation.

    More specifically, `Stochastic` uses `StochasticSpecs` to supply
    key-"specification" pairs. If a specification is an `AbstractSampler`, it
    is **called**, and the returned value is set to the given key. If it is a
    `~pyro.distributions.torch_distribution.TorchDistributionMixin`, it is
    wrapped in a `Sampler` with the key as name, and then called. If the
    specification is none of these, it is passed as-is (like in
    `~functools.partial`).
    """

    __slots__ = 'stochastic_specs'

    if TYPE_CHECKING:
        def __new__(cls: Type[_cls], obj: _T, *args, **kwargs) -> Union[_cls, _T]: ...

    def _init__(self, obj, specs: StochasticSpecs = None,
                capsule: Capsule = None, capsule_args: Iterable[Capsule] = (),
                capsule_kwargs: Mapping[str, Capsule] = frozendict(),
                name: str = None):
        self.stochastic_specs = specs if isinstance(specs, StochasticSpecs) else StochasticSpecs(specs)
        super()._init__(obj, name=name, capsule=capsule, capsule_args=capsule_args, capsule_kwargs=capsule_kwargs)

    def _get_kwargs(self, **kwargs):
        return {**kwargs, **{
            key: (val() if isinstance(val, AbstractSampler) else
                  Sampler(val, name=key)() if isinstance(val, TorchDistributionMixin)
                  else val)
            for key, val in self.stochastic_specs
            if key not in kwargs
        }}

    def _scoped_call(self, *args, **kwargs):
        return super()._scoped_call(*args, **self._get_kwargs(**kwargs))

    def __repr__(self):
        return (f'"{self.stochastic_name}": ' if self.stochastic_name else '') + super().__repr__()
