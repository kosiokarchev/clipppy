from __future__ import annotations

from typing import Iterable, Union

from pyro.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistribution, TorchDistributionMixin
from torch import Size


_size = Union[Size, Iterable[int]]
_Distribution = Union[TorchDistribution, TorchDistributionMixin]


class DistributionWrapper(TorchDistribution):
    """Base class for a distribution that delegates to another one."""

    arg_constraints = {}

    def __init__(self, base_dist: _Distribution, batch_shape: _size = None, event_shape: _size = None, validate_args=None):
        super().__init__(
            Size(batch_shape is not None and batch_shape or base_dist.batch_shape),
            Size(event_shape is not None and event_shape or base_dist.event_shape),
            validate_args=validate_args,
        )
        self.base_dist = base_dist

    @property
    def batch_dim(self):
        return len(self.batch_shape)

    def expand(self, batch_shape, _instance=None):
        new = self.__new__(type(self)) if _instance is None else _instance
        new.base_dist = self.base_dist.expand(Size(batch_shape))
        super(DistributionWrapper, new).__init__(batch_shape, self.event_shape, False)
        new._validate_args = self._validate_args
        return new

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def sample(self, sample_shape=Size()):
        return self.rsample(sample_shape) if self.has_rsample else self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=Size()):
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)

    def cdf(self, value):
        return self.base_dist.cdf(value)

    def icdf(self, value):
        return self.base_dist.icdf(value)

    def entropy(self):
        return self.base_dist.entropy()

    def enumerate_support(self, expand=True):
        return self.base_dist.enumerate_support(expand=expand)
