from __future__ import annotations

from abc import ABC
from math import pi

import scipy.stats
import torch
from pyro.distributions import TorchDistribution
from torch import Size
from torch.distributions.constraints import positive, real

from ..utils import call_nontensor


class BaseITSDistribution(TorchDistribution, ABC):
    has_rsample = True

    def rsample(self, sample_shape=Size()):
        return self.icdf(torch.rand(self.shape(sample_shape)))


class SkewNormal(TorchDistribution):
    arg_constraints = {'loc': real, 'scale': positive, 'alpha': real}
    support = real

    def __init__(self, loc=0., scale=1., alpha=0.,
                 batch_shape=Size(), event_shape=Size(), validate_args=None):
        self.loc = torch.as_tensor(loc)
        self.scale = torch.as_tensor(scale)
        self.alpha = torch.as_tensor(alpha)
        super().__init__(batch_shape, event_shape, validate_args)

    @property
    def delta(self):
        return self.alpha / (1 + self.alpha**2)**0.5

    @property
    def mean(self):
        return self.loc + self.scale * self.delta * (2/pi)**0.5

    @property
    def variance(self):
        return self.scale**2 * (1 - (2/pi) * self.delta**2)

    def log_prob(self, value):
        return call_nontensor(scipy.stats.skewnorm.logpdf, value,
                              loc=self.loc, scale=self.scale, a=self.alpha)

    def cdf(self, value):
        return call_nontensor(scipy.stats.skewnorm.cdf, value,
                              loc=self.loc, scale=self.scale, a=self.alpha)

    def icdf(self, value):
        return call_nontensor(scipy.stats.skewnorm.ppf, value,
                              loc=self.loc, scale=self.scale, a=self.alpha)

    def rsample(self, sample_shape=Size()):
        return call_nontensor(scipy.stats.skewnorm.rvs, size=self.shape(sample_shape),
                              loc=self.loc, scale=self.scale, a=self.alpha)
