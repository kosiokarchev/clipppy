import math
from abc import ABC
from functools import cached_property
from math import inf
from typing import Generic, Optional, TypeVar, Union

import torch
from pyro import distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution as _Distribution
from torch import Size, Tensor
from torch.distributions.constraints import interval


_DT = TypeVar('_DT', bound=_Distribution)
_t = Union[float, Tensor]


def _maybe_item(val: Tensor, *maybe_tensors: _t):
    return val.item() if val.numel() == 1 and not any(map(torch.is_tensor, maybe_tensors)) else val


class ConUnDisMixin(_Distribution, Generic[_DT], ABC):
    constraint_lower: Optional[_t] = -inf
    constraint_upper: Optional[_t] = inf

    @cached_property
    def constraint_range(self):
        return self.constraint_upper - self.constraint_lower

    def constrain(self, lower=None, upper=None):
        if lower is not None:
            self.constraint_lower = lower
        if upper is not None:
            self.constraint_upper = upper
        return self

    # TODO: ConUnDis.__init__
    def __init__(self, *args, constraint_lower=None, constraint_upper=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.constrain(constraint_lower, constraint_upper)

    @property
    def support(self):
        # TODO: constraint=None
        return interval(self.constraint_lower, self.constraint_upper)

    @cached_property
    def lower_prob(self):
        return _maybe_item(
            self.cdf(torch.as_tensor(self.constraint_lower))
            if self.constraint_lower is not None else 0.,
            self.constraint_lower
        )

    @cached_property
    def upper_prob(self):
        return _maybe_item(
            self.cdf(torch.as_tensor(self.constraint_upper)) if self.constraint_upper is not None else 1.,
            self.constraint_upper
        )

    @cached_property
    def constrained_prob(self):
        return self.upper_prob - self.lower_prob

    @cached_property
    def constrained_logprob(self):
        return self.constrained_prob.log() if torch.is_tensor(self.constrained_prob) else math.log(self.constrained_prob)

    has_rsample = True

    def sample(self, sample_shape=Size()):
        return self.rsample(sample_shape)

    def rsample(self, sample_shape=Size()):
        u = self.lower_prob.new_empty(self.shape(sample_shape)).uniform_()
        return torch.where(
            self.lower_prob < self.upper_prob,
            self.icdf(self.lower_prob + self.constrained_prob * u),
            self.constraint_lower + self.constraint_range * u
        )
        return self.icdf(

        )
        # return self.icdf(
        #     torch.distributions.Uniform(self.lower_prob, self.upper_prob).sample(sample_shape)
        # )

    def log_prob(self, value):
        return super().log_prob(value) - self.constrained_logprob


class Normal(ConUnDisMixin[dist.Normal], dist.Normal):
    pass


class _Gamma(dist.Gamma):
    def cdf(self, value):
        return torch.igamma(self.concentration, self.rate*value)

    def icdf(self, value):
        # TODO: igammainv
        d = 1 / (9*self.concentration)
        return self.concentration / self.rate * (1 - d + 2**0.5 * torch.erfinv(2*value-1) * d**0.5)**3


class Gamma(ConUnDisMixin[_Gamma], _Gamma):
    constraint_lower = 0.



__all__ = 'Normal', 'Gamma'
