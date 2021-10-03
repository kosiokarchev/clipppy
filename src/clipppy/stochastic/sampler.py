from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from functools import partial
from itertools import filterfalse
from typing import Any, Callable, ClassVar, ContextManager, Generic, Iterable, Literal, Mapping, Union

import pyro
import torch
from more_itertools import first_true, iterate
from pyro.distributions.torch_distribution import TorchDistributionMixin as _Distribution
from pyro.poutine import infer_config
from torch.distributions.constraints import Constraint

from ..utils import _T, _Tin, _Tout, caller, Sentinel
from ..utils.distributions.extra_dimensions import ExtraIndependent



__api__ = 'AbstractSampler', 'Sampler', 'Param', 'PseudoSampler'
__all__ = 'Sampler', 'Param', 'PseudoSampler'


class AbstractSampler(ABC):
    def __call__(self):
        raise NotImplementedError


_ps_func_t = Union[Callable[[], _Tout], _T]
_ps_call_t = Literal[Sentinel.call, Sentinel.no_call, True, False]
_ps_return_t = Union[_Tout, _T]


@dataclass
class PseudoSampler(AbstractSampler, Generic[_T, _Tout]):
    func: _ps_func_t
    call: _ps_call_t = Sentinel.call

    def __call__(self) -> _ps_return_t:
        return self.func() if self.call in (Sentinel.call, True) else self.func

    def __repr__(self):
        return f'<{type(self).__name__}: {self.func!r}>'


class Context(PseudoSampler):
    # TODO: dataclass: cleverer way to move `plate` to the first place
    def __init__(self, context: ContextManager, func: _ps_func_t, call: _ps_call_t = Sentinel.call):
        self.context = context
        super().__init__(func, call)

    def __call__(self, *args, **kwargs):
        with self.context:
            return super().__call__()


class Effect(PseudoSampler[Any, _Tin], Generic[_Tin, _Tout]):
    def __init__(self, effect: Callable[[_Tin], _Tout], func: Callable[[], _Tin]):
        self.effect = effect
        super().__init__(func)

    def __call__(self) -> _Tout:
        return self.effect(super().__call__())


class UnbindEffect(Effect[torch.Tensor, torch.Tensor]):
    def __init__(self, func, dim=-1):
        super().__init__(partial(torch.unbind, dim=dim), func)


@dataclass
class MultiEffect(AbstractSampler, Generic[_Tout]):
    effect: Callable[..., _Tout]
    arg_funcs: Iterable[Callable[[], Any]]
    kwarg_funcs: Mapping[str, Callable[[], Any]]

    def __init__(self, effect: Callable[..., _Tout], *arg_funcs: Callable[[], Any], **kwarg_funcs: Callable[[], Any]):
        self.effect = effect
        self.arg_funcs = arg_funcs
        self.kwarg_funcs = kwarg_funcs

    def __call__(self) -> _Tout:
        return self.effect(
            *(func for func in self.arg_funcs),
            **{name: func() for name, func in self.kwarg_funcs.items()}
        )


@dataclass
class NamedSampler(AbstractSampler, ABC):
    name: str = None

    def set_name(self, name):
        if name is not None:
            self.name = name
        return self


@dataclass
class NamedPseudoSampler(NamedSampler, PseudoSampler[_T, _Tout], ABC):
    _func: ClassVar[Callable[[str, _ps_return_t], _ps_return_t]]

    def __call__(self, **kwargs):
        return self._func(self.name, super().__call__(), **kwargs)


@dataclass
class Deterministic(NamedPseudoSampler):
    _func = staticmethod(pyro.deterministic)
    event_dim: int = None

    def __call__(self):
        super().__call__(event_dim=self.event_dim)


@dataclass
class Factor(NamedPseudoSampler):
    _func = staticmethod(pyro.factor)


@dataclass
class ConcreteSampler(NamedSampler, ABC):
    init: torch.Tensor = None
    support: Constraint = Sentinel.skip


@dataclass
class _Sampler(ConcreteSampler, ABC):
    d: Union[_Distribution, Callable[[], _Distribution]] = None
    expand_by: Union[torch.Size, Iterable[int]] = torch.Size()
    to_event: int = None
    indep: Union[torch.Size, Iterable[int]] = torch.Size()
    mask: torch.Tensor = None
    infer: Mapping[str, Any] = field(init=False)


class Sampler(_Sampler):
    # TODO: dataclass: cleverer way to move `d` to the first place
    def __init__(self, d: Union[_Distribution, Callable[[], _Distribution]],
                 name: str = _Sampler.name, init: torch.Tensor = _Sampler.init, support: Constraint = _Sampler.support,
                 expand_by: Union[torch.Size, Iterable[int]] = _Sampler.expand_by, to_event: int = _Sampler.to_event,
                 indep: Union[torch.Size, Iterable[int]] = _Sampler.indep,
                 mask: torch.Tensor = None, **kwargs):
        super().__init__(**dict(filterfalse((lambda keyval: keyval[0] in ('self', 'kwargs', '__class__')), locals().items())))
        self.infer = dict(init=self.init, mask=self.mask, support=self.support, **kwargs)

    @property
    def infer_msgr(self):
        return infer_config(config_fn=lambda site: {key: val for key, val in self.infer.items() if val is not Sentinel.skip})

    @property
    def distribution(self) -> _Distribution:
        # Call ``self.d`` until a ``_Distribution`` pops out.
        d = first_true(iterate(caller, self.d), pred=_Distribution.__instancecheck__)
        if self.indep:
            d = ExtraIndependent(d, self.indep)
        return d.expand_by(self.expand_by).to_event(self.to_event)

    def __call__(self):
        with self.infer_msgr:
            return pyro.sample(self.name, self.distribution)


@dataclass
class Param(ConcreteSampler):
    event_dim: int = None

    def __call__(self):
        return pyro.param(self.name, init_tensor=self.init, constraint=self.support, event_dim=self.event_dim)
