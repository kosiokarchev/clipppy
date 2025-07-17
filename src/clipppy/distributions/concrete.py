from __future__ import annotations

from abc import ABC
from typing import Iterable, Generic, TypeVar

import scipy.stats
import torch
from math import pi
from pyro.distributions import TorchDistribution, Uniform, ExpandedDistribution
from torch import Size, Tensor
from torch.distributions import AffineTransform, PowerTransform, Transform
from torch.distributions.constraints import positive, real, interval

from phytorchx import fancy_align, to_tensor
from . import SupportedTransformedDistribution
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


class PowerlawDistribution(SupportedTransformedDistribution):
    def __init__(self, power, low, high):
        self.power, self.low, self.high = fancy_align(power, low, high)

        g = self.power + 1
        lowg = self.low**g
        highg = self.high**g

        super().__init__(Uniform(0, 1), [
            AffineTransform(lowg, highg-lowg),
            PowerTransform(1/g)
        ])

    def expand(self, batch_shape, _instance=None):
        return ExpandedDistribution(self, batch_shape)


class AbstractCDFTransform(Transform):
    codomain = interval(0., 1.)
    bijective = True

    def __init__(self, params: Iterable[Tensor], **kwargs):
        super().__init__(**kwargs)
        self.batch_shape = torch.broadcast_shapes(*map(Tensor.size, params))

    def forward_shape(self, shape):
        return torch.broadcast_shapes(self.batch_shape, shape)

    def inverse_shape(self, shape):
        return torch.broadcast_shapes(self.batch_shape, shape)


class GeneralizedGammaCDFTransform(AbstractCDFTransform):
    domain = positive

    def __init__(self, a: Tensor, p: Tensor, d: Tensor, **kwargs):
        super().__init__((a, p, d), **kwargs)
        self.a, self.p, self.d = a, p, d
        self._lgdp = torch.lgamma(self.d / self.p)
        self._log_norm = (self.p / self.a**self.d).log() - self._lgdp

    def _call(self, x: Tensor) -> Tensor:
        return torch.special.gammainc(self.d/self.p, (x/self.a)**self.p)

    def _inv_call(self, y: Tensor) -> Tensor:
        from phytorch.special.gammainc import gammaincinv
        return self.a * gammaincinv(self.d/self.p, y)**(1/self.p)

    def log_abs_det_jacobian(self, x, y):
        return self._log_norm + (self.d-1) * x.log() - (x/self.a)**self.p


_CDFT = TypeVar('_CDFT', bound=AbstractCDFTransform)


class AbstractCDFTDistribution(SupportedTransformedDistribution, Generic[_CDFT]):
    def __init__(self, cdft: _CDFT):
        self._cdft = cdft
        super().__init__(Uniform(0., 1.).expand(cdft.batch_shape), [cdft.inv])


class GeneralizedGammaDistribution(AbstractCDFTDistribution[GeneralizedGammaCDFTransform]):
    def __init__(self, a, p, d):
        self.a, self.p, self.d = map(to_tensor, (a, p, d))
        super().__init__(GeneralizedGammaCDFTransform(self.a, self.p, self.d))

    @property
    def mode(self) -> torch.Tensor:
        return self.a * ((self.d-1)/self.p)**(1/self.p)

    @property
    def mean(self) -> torch.Tensor:
        return self.a * (torch.lgamma((self.d+1)/self.p) - self._cdft._lgdp).exp()

    @property
    def variance(self) -> torch.Tensor:
        return self.a**2 * (torch.lgamma((self.d+2)/self.p) - self._cdft._lgdp).exp() - self.mean**2
