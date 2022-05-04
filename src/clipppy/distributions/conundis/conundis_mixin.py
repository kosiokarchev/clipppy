from __future__ import annotations

import math
from abc import ABC
from copy import copy
from functools import cached_property, partial
from math import inf
from typing import ClassVar, Generic, Mapping, Optional, Type, TypeVar, Union

import torch
from pyro.distributions.torch_distribution import TorchDistribution as _Distribution
from torch import is_tensor, Size, Tensor
from torch.distributions import ComposeTransform, TransformedDistribution
from torch.distributions.constraints import interval
from typing_extensions import TypeAlias


_DT = TypeVar('_DT', bound=_Distribution)
_t: TypeAlias = Union[float, Tensor]
_constraintT: TypeAlias = Optional[_t]


def _maybe_item(val: Tensor, *maybe_tensors: _t):
    return val.item() if val.numel() == 1 and not any(map(is_tensor, maybe_tensors)) else val


def _tensor_func(_tensor_func, _float_func, val1: _t, val2: _t):
    if is_tensor(val1) and not is_tensor(val2):
        val2 = val1.new_tensor(val2)
    if is_tensor(val2) and not is_tensor(val1):
        val1 = val2.new_tensor(val1)
    return (_tensor_func if any(map(is_tensor, (val1, val2))) else _float_func)(val1, val2)
    

maximum = partial(_tensor_func, torch.fmax, max)
minimum = partial(_tensor_func, torch.fmin, min)


class ConUnDisMixin(_Distribution, Generic[_DT], ABC):
    _concrete: ClassVar[Mapping[Type[_DT]], Union[Type[ConUnDisMixin[_DT]], Type[_DT]]] = {}

    def __init_subclass__(cls, register: Type[_DT] = None, **kwargs):
        super().__init_subclass__(**kwargs)

        if register:
            cls._concrete[register] = cls

    @classmethod
    def new_constrained(
        cls: Type[_CDT], d: _DT,
        constraint_lower: _constraintT = None,
        constraint_upper: _constraintT = None
    ) -> Union[_CDT[_DT], _DT]:
        d = copy(d)
        if isinstance(d, ConUnDisMixin):
            if constraint_lower is None or (d.constraint_lower is not None and constraint_lower < d.constraint_lower):
                constraint_lower = d.constraint_lower
            if constraint_upper is None or (d.constraint_upper is not None and constraint_upper > d.constraint_upper):
                constraint_upper = d.constraint_upper
            # TODO: clean up re-constrained ConUnDis
            for prop in (cls.lower_prob, cls.upper_prob, cls.constrained_prob, cls.constrained_logprob, cls.constraint_range):
                d.__dict__.pop(prop.attrname, None)
            d.constrain(constraint_lower, constraint_upper)
        elif type(d) in cls._concrete:
            d.__class__ = cls._concrete[type(d)]
            d.constrain(constraint_lower, constraint_upper)
        elif isinstance(d, TransformedDistribution):
            t = ComposeTransform(d.transforms).inv
            d.base_dist = cls.new_constrained(d.base_dist, *(
                _maybe_item(t(torch.as_tensor(c)), c) if c is not None else c
                for c in (constraint_lower, constraint_upper)))
        elif hasattr(d, 'base_dist'):
            d.base_dist = cls.new_constrained(d.base_dist, constraint_lower, constraint_upper)
        else:
            raise ValueError(f'Cannot constrain instances of {type(d)} (yet?).')
        return d


    constraint_lower: _constraintT = -inf
    constraint_upper: _constraintT = inf

    @cached_property
    def constraint_range(self):
        return self.constraint_upper - self.constraint_lower

    def constrain(self, lower=None, upper=None):
        s = super().support

        if (_lower := getattr(s, 'lower_bound', None)) is not None:
            lower = _lower if lower is None else maximum(lower, _lower)
        if lower is not None:
            self.constraint_lower = lower

        if (_upper := getattr(s, 'upper_bound', None)) is not None:
            upper = _upper if upper is None else minimum(upper, _upper)
        if upper is not None:
            self.constraint_upper = upper

        return self

    # TODO: ConUnDis.__init__
    def __init__(self, *args, constraint_lower=None, constraint_upper=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.constrain(constraint_lower, constraint_upper)

    @property
    def support(self):
        return interval(self.constraint_lower, self.constraint_upper)

    @cached_property
    def lower_prob(self):
        return _maybe_item(
            super().cdf(torch.as_tensor(self.constraint_lower))
            if self.constraint_lower is not None else 0.,
            self.constraint_lower
        )

    @cached_property
    def upper_prob(self):
        return _maybe_item(
            super().cdf(torch.as_tensor(self.constraint_upper)) if self.constraint_upper is not None else 1.,
            self.constraint_upper
        )

    @cached_property
    def constrained_prob(self):
        return self.upper_prob - self.lower_prob

    @cached_property
    def constrained_logprob(self):
        return self.constrained_prob.log() if is_tensor(self.constrained_prob) else math.log(self.constrained_prob)

    has_rsample = True

    def sample(self, sample_shape=Size()):
        return self.rsample(sample_shape)

    def rsample(self, sample_shape=Size()):
        u = torch.as_tensor(self.lower_prob).new_empty(self.shape(sample_shape)).uniform_()
        return torch.where(
            torch.as_tensor(self.lower_prob < self.upper_prob),
            super().icdf(self.lower_prob + self.constrained_prob * u).clamp_(self.constraint_lower, self.constraint_upper),
            self.constraint_lower + self.constraint_range * u
        )

    def log_prob(self, value):
        return super().log_prob(value) - self.constrained_logprob

    def cdf(self, value):
        return (super().cdf(value) - self.lower_prob) / self.constrained_prob

    def icdf(self, value):
        return super().icdf(self.constrained_prob * value + self.lower_prob)


_CDT = TypeVar('_CDT', bound=ConUnDisMixin)
_ConUnDisT = Union[ConUnDisMixin[_DT], _DT]
