from __future__ import annotations

import torch
from pyro.distributions import constraints, TorchDistribution


__all__ = 'InfiniteUniform', 'SemiInfiniteUniform'


class InfiniteUniform(TorchDistribution):
    """
    A uniform distribution over all real numbers.

    Sampling and `log_prob` always return zeros.
    """
    @property
    def arg_constraints(self):
        return {}

    def rsample(self, sample_shape=torch.Size()):
        return torch.zeros(sample_shape)

    @constraints.dependent_property
    def support(self):
        return constraints.real

    def log_prob(self, value):
        return torch.zeros_like(value)


class SemiInfiniteUniform(InfiniteUniform):
    @constraints.dependent_property
    def support(self):
        return constraints.greater_than_eq(0)

    def log_prob(self, value):
        return torch.where(value < 0., value.new_full((), -float('inf')), value.new_zeros(()))
