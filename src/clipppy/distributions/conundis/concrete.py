from types import new_class

import pyro.distributions as dist
import torch

from .conundis_mixin import ConUnDisMixin


Uniform, Normal, HalfNormal, Exponential, Cauchy, HalfCauchy = (
    new_class(cls.__name__, (ConUnDisMixin[cls], cls), dict(register=cls))
    for cls in (dist.Uniform, dist.Normal, dist.HalfNormal, dist.Exponential,
                dist.Cauchy, dist.HalfCauchy)
)


class _Gamma(dist.Gamma):
    def cdf(self, value):
        return torch.igamma(self.concentration, self.rate*value)

    def icdf(self, value):
        # TODO: igammainv
        d = 1 / (9*self.concentration)
        return self.concentration / self.rate * (1 - d + 2**0.5 * torch.erfinv(2*value-1) * d**0.5)**3


class Gamma(ConUnDisMixin[_Gamma], _Gamma, register=dist.Gamma):
    # TODO: constrained gamma is not exact
    constraint_lower = 0.
