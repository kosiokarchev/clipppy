from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from functools import partial
from itertools import filterfalse
from operator import methodcaller
from typing import Any, Callable, Generic, Iterable, Literal, Mapping, Union

import pyro
import torch
from more_itertools import first_true, iterate
from pyro.distributions.torch_distribution import TorchDistributionMixin as _Distribution
from pyro.poutine import infer_config
from torch.distributions.constraints import Constraint

from ..utils import _T, _Tin, _Tout, Sentinel


__api__ = 'AbstractSampler', 'Sampler', 'Param', 'PseudoSampler'
__all__ = 'Sampler', 'Param', 'PseudoSampler'


class AbstractSampler(ABC):
    def __call__(self):
        raise NotImplementedError


@dataclass
class PseudoSampler(AbstractSampler, Generic[_T, _Tout]):
    func: Union[Callable[[], _Tout], _T]
    call: Literal[Sentinel.call, Sentinel.no_call, True, False] = Sentinel.call

    def __call__(self) -> Union[_Tout, _T]:
        return self.func() if self.call in (Sentinel.call, True) else self.func

    def __repr__(self):
        return f'<{type(self).__name__}: {self.func!r}>'


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
class NamedSampler(AbstractSampler, ABC):
    name: str = None

    def set_name(self, name):
        if name is not None:
            self.name = name
        return self


@dataclass
class ConcreteSampler(NamedSampler, ABC):
    init: torch.Tensor = None
    to_event: int = None
    support: Constraint = Sentinel.skip

    @property
    def event_dim(self):
        return self.to_event


@dataclass
class _Sampler(ConcreteSampler, ABC):
    d: Union[_Distribution, Callable[[], _Distribution]] = None
    expand_by: Union[torch.Size, Iterable[int]] = torch.Size()
    mask: torch.Tensor = None
    infer: Mapping[str, Any] = field(init=False)


class Sampler(_Sampler):
    # TODO: cleverer way to move `d` to the first place
    def __init__(self, d: _Distribution,
                 name: str = None, init: torch.Tensor = None, to_event: int = None, support: Constraint = Sentinel.skip,
                 expand_by: Union[torch.Size, Iterable[int]] = torch.Size(), mask: torch.Tensor = None,
                 **kwargs):
        super().__init__(**dict(filterfalse((lambda keyval: keyval[0] in ('self', 'kwargs', '__class__')), locals().items())))
        self.infer = dict(init=self.init, mask=self.mask, support=self.support, **kwargs)

    @property
    def infer_msgr(self):
        return infer_config(config_fn=lambda site: {key: val for key, val in self.infer.items() if val is not Sentinel.skip})

    @property
    def distribution(self) -> _Distribution:
        # Call ``self.d`` until a ``_Distribution`` pops out.
        return first_true(iterate(methodcaller('__call__'), self.d),
                          pred=_Distribution.__instancecheck__)

    def __call__(self):
        with self.infer_msgr:
            return pyro.sample(self.name, self.distribution.expand_by(self.expand_by).to_event(self.event_dim))


class Param(ConcreteSampler):
    def __call__(self):
        return pyro.param(self.name, init_tensor=self.init, constraint=self.support, event_dim=self.event_dim)
