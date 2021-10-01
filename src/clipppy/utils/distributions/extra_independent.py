from typing import Union

import torch
from pyro.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistributionMixin, TorchDistribution
from torch import Tensor


_size = Union[torch.Size, list[int], tuple[int, ...]]
_Distribution = Union[TorchDistribution, TorchDistributionMixin]


class DistributionWrapper(TorchDistribution):
    """Base class for a distribution that delegates to another one."""

    arg_constraints = {}

    def __init__(self, base_dist: _Distribution, batch_shape: _size = None, event_shape: _size = None, validate_args=None):
        super().__init__(
            batch_shape is not None and batch_shape or base_dist.batch_shape,
            event_shape is not None and event_shape or base_dist.event_shape,
            validate_args=validate_args,
        )
        self.base_dist = base_dist

    @property
    def batch_dim(self):
        return len(self.batch_shape)

    def expand(self, batch_shape, _instance=None):
        new = self.__new__(type(self)) if _instance is None else _instance
        new.base_dist = self.base_dist.expand(torch.Size(batch_shape))
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

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)

    def entropy(self):
        return self.base_dist.entropy()

    def enumerate_support(self, expand=True):
        return self.base_dist.enumerate_support(expand=expand)


class ExtraDimensions(DistributionWrapper):
    def __init__(self, base_dist: _Distribution, extra_shape: _size, batch_shape: _size = None, event_shape: _size = None, validate_args=None):
        super().__init__(base_dist, batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

        self.extra_shape = torch.Size(extra_shape)
        self.extra_dim = len(self.extra_shape)
        self.extra_end = -len(self.base_dist.shape())
        self.extra_start = self.extra_end - self.extra_dim
        self.extra_loc = -self.base_dist.event_dim

    @property
    def sample_dim(self):
        return len(self.extra_shape)

    def expand(self, batch_shape, _instance=None):
        new = super().expand(batch_shape, _instance)
        new.sample_shape = self.extra_shape
        return new

    def roll_to_right(self, value: Tensor):
        return value.permute(
            tuple(range(value.ndim+self.extra_start))
            + tuple(range(self.extra_end, self.extra_loc))
            + tuple(range(self.extra_start, self.extra_end))
            + tuple(range(self.extra_loc, 0))
        )

    def sample(self, sample_shape: _size = torch.Size()):
        return self.roll_to_right(
            self.base_dist.sample(sample_shape + self.extra_shape)
        )

    def rsample(self, sample_shape: _size = torch.Size()):
        return self.roll_to_right(
            self.base_dist.rsample(sample_shape + self.extra_shape)
        )

    def entropy(self):
        return self.base_dist.entropy()


class ExtraBatched(ExtraDimensions):
    def __init__(self, base_dist: _Distribution, extra_shape: _size, validate_args=None):
        super().__init__(base_dist, extra_shape=extra_shape, batch_shape=base_dist.batch_shape + extra_shape, validate_args=validate_args)


class ExtraIndependent(ExtraDimensions):
    """
    Add more event dimensions without consuming existing batch dimensions.

    Useful in order to treat multiple samples as a dimension in an event and
    sum them in the probability and to allow proper plating.
    """

    def __init__(self, base_dist: _Distribution, extra_shape: _size, validate_args=None):
        super().__init__(base_dist, extra_shape=extra_shape, event_shape=extra_shape + base_dist.event_shape, validate_args=validate_args)

    def log_prob(self, value):
        return super().log_prob(value).sum(range(-self.extra_dim, 0))
