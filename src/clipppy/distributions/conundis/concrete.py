from __future__ import annotations

from types import new_class

import pyro.distributions as dist
import torch

from .conundis_mixin import ConUnDisMixin

try:
    # TODO: dependence of phytorch
    from phytorch.special.gammainc import gammainccinv, gammaincinv
except ImportError:
    from functools import partial
    from scipy import special as sp
    from ...utils import call_nontensor

    gammaincinv = partial(call_nontensor, sp.gammaincinv)
    gammainccinv = partial(call_nontensor, sp.gammainccinv)


Uniform, Normal, HalfNormal, Exponential, Cauchy, HalfCauchy = (
    new_class(cls.__name__, (ConUnDisMixin[cls], cls), dict(register=cls),
              lambda ns: ns.update({'__module__': __name__}))
    for cls in (dist.Uniform, dist.Normal, dist.HalfNormal, dist.Exponential,
                dist.Cauchy, dist.HalfCauchy)
)


class _Gamma(dist.Gamma):
    def cdf(self, value):
        return torch.special.gammainc(self.concentration, self.rate * value)

    def icdf(self, value):
        return gammaincinv(self.concentration, value) / self.rate


class Gamma(ConUnDisMixin[_Gamma], _Gamma, register=dist.Gamma):
    # TODO: constrained gamma uses SciPy
    constraint_lower = 0.


class _InverseGamma(dist.InverseGamma):
    def cdf(self, value):
        return torch.special.gammaincc(self.concentration, self.rate / value)

    def icdf(self, value):
        return self.rate / gammainccinv(self.concentration, value)


class InverseGamma(ConUnDisMixin[_InverseGamma], _InverseGamma, register=dist.InverseGamma):
    # TODO: constrained inverse gamma uses SciPy
    constraint_lower = 0.
