from __future__ import annotations

from copy import copy
from typing import Union

import torch
from pyro.distributions.torch_distribution import (ExpandedDistribution, MaskedDistribution, TorchDistribution,
                                                   TorchDistributionMixin)
from pyro.distributions.util import broadcast_shape
from torch.distributions import Independent


class EventMaskedDistribution(MaskedDistribution):
    def __init__(self, base_dist, mask):
        if isinstance(mask, bool):
            self._mask = mask
        else:
            self._mask = mask.bool()
        self.base_dist = base_dist
        super(MaskedDistribution, self).__init__(base_dist.batch_shape, base_dist.event_shape)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(EventMaskedDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.base_dist = self.base_dist.expand(batch_shape)
        new._mask = self._mask
        super(MaskedDistribution, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        if self._mask is False:
            shape = broadcast_shape(self.base_dist.batch_shape,
                                    value.shape[:value.dim() - self.event_dim])
            return torch.zeros((), device=value.device).expand(shape)
        if self._mask is True:
            return self.base_dist.log_prob(value)

        assert len(self.event_shape) >= len(self._mask.shape)

        d = self.base_dist
        full_shape = d.shape()
        event_shape = full_shape[len(d.batch_shape) + self._mask.ndim:]

        while isinstance(d, Independent) or isinstance(d, ExpandedDistribution):
            d = d.base_dist

        d: Union[TorchDistribution, TorchDistributionMixin] \
            = copy(d)
        d._batch_shape = d.shape()[:-len(event_shape)]
        d._event_shape = event_shape
        ret = d.log_prob(value)[..., self._mask]

        return ret.sum(-1)
