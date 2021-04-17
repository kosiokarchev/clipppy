from functools import partialmethod

import torch
from pyro import distributions as dist

from clipppy.stochastic.sampler import Sampler


__all__ = 'InfiniteUniform', 'SemiInfiniteUniform', 'InfiniteSampler', 'SemiInfiniteSampler'


class InfiniteUniform(dist.TorchDistribution):
    @property
    def arg_constraints(self):
        return {}

    def rsample(self, sample_shape=torch.Size()):
        return torch.zeros(sample_shape)

    @dist.constraints.dependent_property
    def support(self):
        return dist.constraints.real

    def log_prob(self, value):
        return torch.zeros_like(value)


class SemiInfiniteUniform(InfiniteUniform):
    @dist.constraints.dependent_property
    def support(self):
        return dist.constraints.positive

    def log_prob(self, value):
        return torch.where(value < 0., value.new_full((), -float('inf')), value.new_zeros(()))


class InfiniteSampler(Sampler):
    __init__ = partialmethod(Sampler.__init__, InfiniteUniform())


class SemiInfiniteSampler(Sampler):
    __init__ = partialmethod(Sampler.__init__, SemiInfiniteUniform())
