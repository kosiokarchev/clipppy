import torch
from torch import Tensor

from .wrapper import _Distribution, _size, DistributionWrapper


class ExtraDimensions(DistributionWrapper):
    def __init__(self, base_dist: _Distribution, extra_shape: _size, batch_shape: _size = None, event_shape: _size = None, validate_args=None):
        super().__init__(base_dist, batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

        self.extra_shape = torch.Size(extra_shape)
        self.extra_dim = len(self.extra_shape)

    @property
    def sample_dim(self):
        return len(self.extra_shape)

    def expand(self, batch_shape, _instance=None):
        new = super().expand(batch_shape, _instance)
        new.extra_shape = self.extra_shape
        new.extra_dim = self.extra_dim
        return new

    def roll_to_right(self, value: Tensor):
        extra_loc = value.ndim - self.base_dist.event_dim
        return value.permute(
            tuple(range(self.extra_dim, extra_loc))
            + tuple(range(self.extra_dim))
            + tuple(range(extra_loc, value.ndim))
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
