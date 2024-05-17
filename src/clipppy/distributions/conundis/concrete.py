from __future__ import annotations

from types import new_class
from typing import TYPE_CHECKING

import pyro.distributions as dist
import torch

from .conundis_mixin import ConUnDisMixin

if TYPE_CHECKING:
    # TODO: dependence of phytorch
    from phytorch.special.gammainc import gammainccinv, gammaincinv


def __getattr__(name):
    if name in ('gammainccinv', 'gammaincinv'):
        try:
            import phytorch.special.gammainc
            res = getattr(phytorch.special, name)
        except (ImportError, AttributeError):
            from functools import partial
            from scipy import special as sp
            from ...utils import call_nontensor

            res = partial(call_nontensor, getattr(sp, name))

        globals()[name] = res
        return res
    else:
        raise AttributeError(name)


Uniform, Normal, HalfNormal, Exponential, Cauchy, HalfCauchy, Pareto = (
    new_class(cls.__name__, (ConUnDisMixin[cls], cls), dict(register=cls),
              lambda ns: ns.update({'__module__': __name__}))
    for cls in (dist.Uniform, dist.Normal, dist.HalfNormal, dist.Exponential,
                dist.Cauchy, dist.HalfCauchy, dist.Pareto)
)


class _Gamma(dist.Gamma):
    def cdf(self, value):
        return torch.special.gammainc(self.concentration, self.rate * value)

    def icdf(self, value):
        return gammaincinv(self.concentration, value) / self.rate


class Gamma(ConUnDisMixin[_Gamma], _Gamma, register=dist.Gamma):
    constraint_lower = 0.


class _InverseGamma(dist.InverseGamma):
    def cdf(self, value):
        return torch.special.gammaincc(self.concentration, self.rate / value)

    def icdf(self, value):
        return self.rate / gammainccinv(self.concentration, value)


class InverseGamma(ConUnDisMixin[_InverseGamma], _InverseGamma, register=dist.InverseGamma):
    constraint_lower = 0.
