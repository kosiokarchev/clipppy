from functools import partial
from types import new_class

import pyro.distributions as dist
import torch
from scipy import special as sp

from .conundis_mixin import ConUnDisMixin


def _call_sp(func, *args):
    if any((torch.is_tensor(arg) and arg.requires_grad) for arg in args):
        raise NotImplementedError

    return func(*(
        a.cpu() if torch.is_tensor(a) else a
        for a in args
    )).to(next(filter(torch.is_tensor, args)))


gammaincinv = partial(_call_sp, sp.gammaincinv)
gammainccinv = partial(_call_sp, sp.gammainccinv)


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
