from __future__ import annotations

from abc import ABC
from copy import copy
from functools import cached_property, partial
from math import inf
from typing import Generic, Optional, TypeVar, Union, Callable

import torch
from torch import is_tensor, Size, Tensor
from torch.distributions import ComposeTransform, TransformedDistribution
from torch.distributions.constraints import interval
from typing_extensions import TypeAlias, Self

from ..constrained import ConstrainedDistribution, _DT, _t


_constraintT: TypeAlias = Optional[_t]


def _maybe_item(val: Tensor, *maybe_tensors: _t):
    return val.item() if val.numel() == 1 and not any(map(is_tensor, maybe_tensors)) else val


def _tensor_func(
    _tensor_func: Callable[[Tensor, Tensor], Tensor],
    _float_func: Callable[[float, float], float],
    val1: _t, val2: _t
) -> _t:
    if is_tensor(val1) and not is_tensor(val2):
        val2 = val1.new_tensor(val2)
    if is_tensor(val2) and not is_tensor(val1):
        val1 = val2.new_tensor(val1)
    return (_tensor_func if any(map(is_tensor, (val1, val2))) else _float_func)(val1, val2)
    

maximum = partial(_tensor_func, torch.fmax, max)
minimum = partial(_tensor_func, torch.fmin, min)


def _check_and_choose(val1: _constraintT, val2: _constraintT, choose: Callable[[_t, _t], _t]):
    return (
        val1 if val2 is None else
        val2 if val1 is None else
        choose(val1, val2)
    )


class ConUnDisMixin(ConstrainedDistribution, Generic[_DT], ABC, final_constrained=True):
    @classmethod
    def new_constrained(
        cls, d: _DT,
        constraint_lower: _constraintT = None,
        constraint_upper: _constraintT = None
    ) -> Union[Self, _DT]:
        d = copy(d)

        # TODO: that's a hack for LeftIndependent
        if hasattr(d, 'prepare_sample'):
            constraint_lower = d.prepare_sample(constraint_lower)
            constraint_upper = d.prepare_sample(constraint_upper)

        if isinstance(d, ConUnDisMixin):
            d.reconstrain(
                _check_and_choose(constraint_lower, d.constraint_lower, maximum),
                _check_and_choose(constraint_upper, d.constraint_upper, minimum))
        elif isinstance(d, TransformedDistribution) and type(d) not in cls._concrete:
            t = ComposeTransform(d.transforms).inv
            d.base_dist = cls.new_constrained(d.base_dist, *(
                _maybe_item(t(torch.as_tensor(c)), c) if c is not None else c
                for c in (constraint_lower, constraint_upper)))
        else:
            d = super().new_constrained(d, constraint_lower, constraint_upper)

        return d


    constraint_lower: _constraintT = -inf
    constraint_upper: _constraintT = inf

    @cached_property
    def constraint_shape(self) -> Size:
        return torch.broadcast_shapes(*(
            getattr(v, 'shape', Size())
            for v in (self.constraint_lower, self.constraint_upper)
        ))

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
            super().cdf(torch.as_tensor(self.constraint_lower)),
            self.constraint_lower
        ) if self.constraint_lower not in (None, -inf) else 0.

    @cached_property
    def upper_prob(self):
        return _maybe_item(
            super().cdf(torch.as_tensor(self.constraint_upper)),
            self.constraint_upper
        ) if self.constraint_upper not in (None, inf) else 1.

    @cached_property
    def constrained_prob(self):
        return self.upper_prob - self.lower_prob

    has_rsample = True

    def sample(self, sample_shape=Size()):
        return ConstrainedDistribution.sample(self, sample_shape)

    def rsample(self, sample_shape=Size()):
        u = torch.as_tensor(self.lower_prob).new_empty(self.shape(sample_shape)).uniform_()
        return torch.where(
            torch.as_tensor(self.lower_prob < self.upper_prob),
            super().icdf(self.lower_prob + self.constrained_prob * u).clamp_(self.constraint_lower, self.constraint_upper),
            self.constraint_lower + self.constraint_range * u
        )

    def cdf(self, value):
        return (super().cdf(value) - self.lower_prob) / self.constrained_prob

    def icdf(self, value):
        return super().icdf(self.constrained_prob * value + self.lower_prob)


_CDT = TypeVar('_CDT', bound=ConUnDisMixin)
_ConUnDisT = Union[ConUnDisMixin[_DT], _DT]
