from __future__ import annotations

import torch
from torch import Tensor

from .wrapper import _Distribution, _size, DistributionWrapper


_sizeify = lambda size: None if size is None else torch.Size(size)


class ExtraDimensions(DistributionWrapper):
    def __init__(self, base_dist: _Distribution, extra_shape: _size, batch_shape: _size = None, event_shape: _size = None, validate_args=None):
        super().__init__(base_dist, batch_shape=_sizeify(batch_shape), event_shape=_sizeify(event_shape), validate_args=validate_args)

        self.extra_shape = _sizeify(extra_shape)
        self.extra_dim = len(self.extra_shape)

    @property
    def sample_dim(self):
        return len(self.extra_shape)

    def expand(self, batch_shape, _instance=None):
        new = super().expand(_sizeify(batch_shape), _instance)
        new.extra_shape = _sizeify(self.extra_shape)
        new.extra_dim = self.extra_dim
        return new

    def extra_dims(self, value: Tensor):
        extra_loc = value.ndim - self.base_dist.event_dim
        return tuple(range(extra_loc - self.extra_dim, extra_loc))

    def roll_to_right(self, value: Tensor) -> Tensor:
        return value.movedim(tuple(range(self.extra_dim)), self.extra_dims(value))

    def roll_to_left(self, value: Tensor) -> Tensor:
        return value.movedim(self.extra_dims(value), tuple(range(self.extra_dim)))

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        return self.roll_to_right(self.base_dist.rsample(self.extra_shape + sample_shape))

    def log_prob(self, value: Tensor) -> Tensor:
        return super().log_prob(self.roll_to_left(value.expand(
            value.shape[:value.ndim - (self.batch_dim+self.event_dim)] + self.batch_shape + self.event_shape)
        )).movedim(
            tuple(range(self.extra_dim)), tuple(range(-self.extra_dim, 0)))

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
        return super().log_prob(value).sum(tuple(range(-self.extra_dim, 0)))
