from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from itertools import filterfalse
from typing import (
    Any, Callable, ClassVar, ContextManager, Generic, Iterable, Literal,
    Mapping, Optional, Type, TypeVar, Union)

import pyro
import torch
from more_itertools import first_true, iterate
from pyro.distributions.torch_distribution import TorchDistributionMixin as _Distribution
from torch import Tensor
from torch.distributions.constraints import Constraint
from typing_extensions import TypeAlias

from ..utils import _T, _Tin, _Tout, caller, Sentinel
from ..distributions.extra_dimensions import LeftIndependent

__all__ = (
    'AbstractSampler', 'NamedSampler', 'ConcreteSampler', 'PseudoSampler', 'NamedPseudoSampler',
    'Context', 'Effect', 'UnbindEffect', 'UnsqueezeEffect', 'MovedimEffect',
    'Sampler', 'Param', 'Deterministic', 'Factor')


T = TypeVar('T')


class AbstractSampler(ABC):
    """The base class for "samplers": objects that dynamically (or not) generate values in a `Stochastic` context."""

    @abstractmethod
    def __call__(self):
        """
        An `AbstractSampler` is called when `Stochastic` encounters one as a
        "specification", and it is this method that has to be overridden in
        subclasses to supply the desired value.
        """


_ps_func_t: TypeAlias = Union[Callable[[], _Tout], _T]
_ps_call_t: TypeAlias = Literal[Sentinel.call, Sentinel.no_call, True, False]
_ps_return_t: TypeAlias = Union[_Tout, _T]


@dataclass
class PseudoSampler(Generic[_T, _Tout], AbstractSampler):
    func_or_val: _ps_func_t
    call: _ps_call_t = Sentinel.call

    def __call__(self) -> _ps_return_t:
        return (
            self.func_or_val()
            if self.call in (Sentinel.call, True) and callable(self.func_or_val)
            else self.func_or_val)

    def __repr__(self):
        return f'<{type(self).__name__}: {self.func_or_val!r}>'


class Context(PseudoSampler):
    # TODO: dataclass: cleverer way to move `plate` to the first place
    def __init__(self, context: ContextManager,
                 func_or_val: _ps_func_t, call: _ps_call_t = PseudoSampler.call):
        self.context = context
        super().__init__(func_or_val, call)

    def __call__(self, *args, **kwargs):
        with self.context:
            return super().__call__()


class Effect(PseudoSampler[_Tin, _Tin], Generic[_Tin, _Tout]):
    # TODO: dataclass: cleverer way to move `effect` to the first place
    def __init__(self, effect: Callable[[_Tin], _Tout],
                 func_or_val: _ps_func_t, call: _ps_call_t = PseudoSampler.call):
        self.effect = effect
        super().__init__(func_or_val, call)

    def __call__(self) -> _Tout:
        return self.effect(super().__call__())


class UnbindEffect(Effect[Tensor, Tensor]):
    def __init__(self, func_or_val: _ps_func_t, call: _ps_call_t = PseudoSampler.call, dim=-1):
        super().__init__(partial(torch.unbind, dim=dim), func_or_val, call)


class UnsqueezeEffect(Effect[Tensor, Tensor]):
    def __init__(self, func_or_val: _ps_func_t, call: _ps_call_t = PseudoSampler.call, dim=-1):
        super().__init__(partial(torch.unsqueeze, dim=dim), func_or_val, call)


class MovedimEffect(Effect[Tensor, Tensor]):
    def __init__(self, func_or_val: _ps_func_t, call: _ps_call_t = PseudoSampler.call, source=-1, destination=0):
        super().__init__(partial(torch.movedim, source=source, destination=destination), func_or_val, call)


# TODO: MultiEffect: a lightweight Stochastic alternative?
# @dataclass
# class MultiEffect(AbstractSampler, Generic[_Tout]):
#     effect: Callable[..., _Tout]
#     arg_funcs: Iterable[Callable[[], Any]]
#     kwarg_funcs: Mapping[str, Callable[[], Any]]
#
#     def __init__(self, effect: Callable[..., _Tout], *arg_funcs: Callable[[], Any], **kwarg_funcs: Callable[[], Any]):
#         self.effect = effect
#         self.arg_funcs = arg_funcs
#         self.kwarg_funcs = kwarg_funcs
#
#     def __call__(self) -> _Tout:
#         return self.effect(
#             *(func() for func in self.arg_funcs),
#             **{name: func() for name, func in self.kwarg_funcs.items()}
#         )


@dataclass
class NamedSampler(AbstractSampler, ABC):
    """A base class for samplers with a name.

    These include `Sampler`, `Param`, `Deterministic`, and `Factor` that
    correspond to the respective `pyro.primitives`.
    """

    name: str = None
    """The name of the sampler, used e.g. in `pyro.sample <pyro.primitives.sample>` and
    `pyro.param <pyro.primitives.param>`."""

    def set_name(self: T, name: Optional[str]) -> T:
        r"""Set ``self.``\ `~NamedSampler.name` and return ``self``.

        Parameters
        ----------
        name: str or None
            Set ``self.``\ `~NamedSampler.name` to this, but only if it (the
            passed in argument) is not `None`.

        Returns
        -------
        NamedSampler
            ``self``
        """
        if name is not None:
            self.name = name
        return self

    _subclasses: ClassVar[set[Type[NamedSampler]]] = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        cls._subclasses.add(cls)


@dataclass
class NamedPseudoSampler(NamedSampler, PseudoSampler[_T, _Tout], ABC):
    """A `PseudoSampler` with a `name <NamedSampler>`.

    Processes the value returned by `PseudoSampler.__call__` through a
    function `~NamedPseudoSampler._func` that is given the `~NamedSampler.name`
    as first argument and the value as second.
    """

    _func: ClassVar[Callable[[str, _ps_return_t], _ps_return_t]]
    """A function to process the value before returning.
    
    It is called as :python:`self._func(self.name, val, **kwargs)`. Subclasses
    can override this to e.g. `pyro.deterministic <pyro.primitives.deterministic>`
    and `pyro.factor <pyro.primitives.factor>`.
    """

    def __call__(self, **kwargs):
        ret = super().__call__()
        return self._func(self.name, ret, **kwargs)


@dataclass
class Deterministic(NamedPseudoSampler):
    """A `PseudoSampler` that records the value via `pyro.deterministic <pyro.primitives.deterministic>`."""

    _func = staticmethod(pyro.deterministic)

    event_dim: int = None
    """The ``event_dim`` parameter to `pyro.deterministic <pyro.primitives.deterministic>`."""

    def __call__(self):
        return super().__call__(event_dim=self.event_dim)


@dataclass
class Factor(NamedPseudoSampler):
    """A `PseudoSampler` that records the value as a `pyro.factor <pyro.primitives.factor>`."""

    _func = staticmethod(pyro.factor)


@dataclass
class ConcreteSampler(NamedSampler, ABC):
    """A base class for "true" samplers, i.e. that provide a value from |Pyro|.

    Namely, `Param` and `Sampler`.
    """

    init: torch.Tensor = Sentinel.skip
    support: Constraint = Sentinel.skip


@dataclass
class _Sampler(ConcreteSampler, ABC):
    expand_by: Union[torch.Size, Iterable[int]] = torch.Size()
    to_event: int = None
    indep: Union[torch.Size, Iterable[int]] = torch.Size()
    mask: torch.Tensor = Sentinel.skip


_Sampler_dT: TypeAlias = Union[_Distribution, Callable[[], '_Sampler_dT']]


class Sampler(_Sampler):
    """Represents a `pyro.sample <pyro.primitives.sample>` statement."""

    d: _Sampler_dT
    r""": A\ `( callable that returns a)* <https://regex101.com/r/1QwDsK>`_
    `distribution <pyro.distributions.torch_distribution.TorchDistributionMixin>`
    to be passed to `pyro.sample <pyro.primitives.sample>`."""

    infer: Mapping[str, Any]

    # TODO: dataclass: cleverer way to move `d` to the first place
    def __init__(self, d: _Sampler_dT,
                 # NamedSampler
                 name: str = _Sampler.name,
                 # ConcreteSampler
                 init: torch.Tensor = _Sampler.init, support: Constraint = _Sampler.support,
                 # _Sampler
                 expand_by: Union[torch.Size, Iterable[int]] = _Sampler.expand_by,
                 to_event: int = _Sampler.to_event,
                 indep: Union[torch.Size, Iterable[int]] = _Sampler.indep,
                 mask: torch.Tensor = _Sampler.mask, **kwargs):
        super().__init__(**dict(filterfalse((lambda keyval: keyval[0] in ('self', 'd', 'kwargs', '__class__')), locals().items())))
        self.d = d
        self.infer = dict(init=self.init, mask=self.mask, support=self.support, **kwargs)

    @property
    def infer_dict(self) -> dict[str, Any]:
        r"""A `dict` to be used as a sample site's ``config`` attribute.

        It is constructed from ``self.``\ `~Sampler.infer` by filtering out
        `Sentinel.skip` values.
        """
        return {key: val for key, val in self.infer.items() if val is not Sentinel.skip}

    @property
    def distribution(self) -> _Distribution:
        r"""The distribution from which to sample.

        This is the result of repeatedly calling ``self.``\ `Sampler.d` (zero
        or more times) until a `distribution <pyro.distributions.torch_distribution.TorchDistributionMixin>`
        pops out.
        """
        d = first_true(iterate(caller, self.d), pred=_Distribution.__instancecheck__)
        if self.expand_by:
            d = d.expand_by(self.expand_by)
        if self.indep:
            d = LeftIndependent(d.expand_by(indep := torch.Size(self.indep)), len(indep))
        return d.to_event(self.to_event)

    def __call__(self):
        # callables in self.d should not be wrapped in self.infer_msgr
        return pyro.sample(self.name, self.distribution, infer=self.infer_dict)


@dataclass
class Param(ConcreteSampler):
    """Represents a `pyro.param <pyro.primitives.param>` statement."""

    event_dim: int = None
    """The ``event_dim`` argument to `pyro.param <pyro.primitives.param>`."""

    def __call__(self):
        r"""Call `pyro.param <pyro.primitives.param>` with suitable arguments.

        Passes in
            - ``self.``\ `~NamedSampler.name`,
            - ``self.``\ `~ConcreteSampler.init` as ``init_tensor``,
            - ``self.``\ `~ConcreteSampler.support` as ``constraint``, and
            - ``self.``\ `~Param.event_dim`.

        Returns
        -------
        torch.Tensor
            The result of the `~pyro.primitives.param` statement.
        """
        return pyro.param(self.name, init_tensor=self.init, constraint=self.support, event_dim=self.event_dim)
