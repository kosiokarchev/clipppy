from __future__ import annotations

import math
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, TypeVar, Union, Type, ClassVar, MutableMapping

from pyro.distributions.torch_distribution import TorchDistribution, TorchDistributionMixin
from torch import is_tensor, Size, Tensor
from typing_extensions import Self, TypeAlias, ParamSpec

from .utils import process_log_prob
from .wrapper import unwrap


_t = TypeVar('_t', bound=Union[float, Tensor])
_DT = TypeVar('_DT', bound=Union[TorchDistribution, TorchDistributionMixin])
_tDT: TypeAlias = Type[_DT]
_PS = ParamSpec('_PS')


class ConstrainedDistribution(TorchDistribution, Generic[_DT, _PS], ABC):
    _concrete: ClassVar[MutableMapping[_tDT], Union[Type[Self], _tDT]]

    def __init_subclass__(cls, final_constrained=False, register: _tDT = None, **kwargs):
        super().__init_subclass__(**kwargs)

        if final_constrained:
            cls._concrete = {}

        if register:
            cls._concrete[register] = cls

    @classmethod
    def new_constrained(cls, d: _DT, *args: _PS.args, **kwargs: _PS.kwargs) -> Union[Self, _DT]:
        if type(d) in cls._concrete:
            d.__class__ = cls._concrete[type(d)]
            d: Union[Self, _DT]
            d.constrain(*args, **kwargs)
            if len(d.constraint_shape) > len(d.batch_shape + d.event_shape):
                raise ValueError(f'constraint shape {d.constraint_shape}'
                                 ' has more dimensions than the sample shape'
                                 f' {d.batch_shape + d.event_shape}')
        elif hasattr(d, 'base_dist'):
            d.base_dist = cls.new_constrained(d.base_dist, *args, **kwargs)
        else:
            raise ValueError(f'Cannot constrain instances of {type(d)} (yet?).')

        return d


    @abstractmethod
    def constrain(self, *args: _PS.args, **kwargs: _PS.kwargs) -> Self: ...

    def clean_up(self):
        for typ in type(self).mro():
            if issubclass(typ, ConstrainedDistribution):
                for key, val in vars(typ).items():
                    if isinstance(val, cached_property):
                        self.__dict__.pop(key, None)

    def reconstrain(self, *args: _PS.args, **kwargs: _PS.kwargs) -> Self:
        self.clean_up()
        return self.constrain(*args, **kwargs)


    @property
    @abstractmethod
    def constrained_prob(self) -> _t: ...

    @cached_property
    def constrained_log_prob(self) -> _t:
        return cprob.log() if is_tensor(cprob := self.constrained_prob) else math.log(cprob)

    def log_prob(self, value):
        return super().log_prob(value) - self.constrained_log_prob

    @property
    @abstractmethod
    def constraint_shape(self) -> Size: ...


def constrained_log_prob(d: _DT) -> _t:
    *wrappers, d = unwrap(d, ConstrainedDistribution)
    if isinstance(d, ConstrainedDistribution):
        clp = d.constrained_log_prob
        for w in reversed(wrappers):
            clp = process_log_prob(w, clp)
        return clp
    else:
        return 0.
