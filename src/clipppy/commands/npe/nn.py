from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Generic, TYPE_CHECKING, TypeVar

import attr
import torch.autograd
from pyro.distributions import MultivariateNormal, Normal
from torch import Tensor
from torch.distributions import biject_to
from torch.distributions.constraints import Constraint, corr_cholesky, positive, real

from ..npe._typing import Distribution, TransformedDistribution
from ...sbi.nn import _HeadOoutT, _HeadPoutT, BaseSBITail


_DistributionT = TypeVar('_DistributionT', bound=Distribution)


@dataclass
class NPEResult(Generic[_DistributionT]):
    # A namedtuple-like that does not get flattened by pytree (+ caching)
    theta: Tensor
    q: _DistributionT
    requires_grad: bool = False

    def __iter__(self):
        return iter((self.theta, self.q))

    @cached_property
    def log_prob(self):
        return self.q.log_prob(self.theta.requires_grad_(self.requires_grad))

    @cached_property
    def log_prob_grad(self):
        return torch.autograd.grad(
            self.log_prob, self.theta, torch.ones_like(self.log_prob),
            create_graph=True
        )[0]


class BaseNPETail(BaseSBITail[_HeadPoutT, _HeadOoutT, NPEResult[_DistributionT]], Generic[_HeadPoutT, _HeadOoutT, _DistributionT], ABC):
    pass


class NPETail(BaseNPETail[Tensor, _HeadOoutT, _DistributionT], Generic[_HeadOoutT, _DistributionT], ABC):
    @abstractmethod
    def forward(self, theta: Tensor, x: _HeadOoutT, **kwargs) -> NPEResult[_DistributionT]: ...

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class NormalTail(NPETail[Tensor, TransformedDistribution]):
    ndim: int
    constraint: Constraint = real

    @cached_property
    def event_size(self):
        return 2 * self.ndim

    @cached_property
    def biject_to_constraint(self):
        return biject_to(self.constraint)

    def extract_loc(self, x: Tensor) -> Tensor:
        return x[..., :self.ndim]

    def extract_scale(self, x: Tensor) -> Tensor:
        return self.biject_to_positive(x[..., self.ndim:2 * self.ndim])

    def _transform(self, d: Distribution):
        return TransformedDistribution(d, [self.biject_to_constraint])

    def _get_dist(self, x: Tensor):
        return Normal(self.extract_loc(x), self.extract_scale(x))

    def forward(self, theta: Tensor, x: Tensor, **kwargs) -> NPEResult[TransformedDistribution]:
        return NPEResult(theta, self._transform(self._get_dist(x)), **kwargs)

    if TYPE_CHECKING:
        __call__ = forward



class MVNTail(NormalTail):
    biject_to_positive: ClassVar = biject_to(positive)
    biject_to_corr_cholesky: ClassVar = biject_to(corr_cholesky)

    @cached_property
    def event_size(self):
        return 2 * self.ndim + (self.ndim * (self.ndim - 1)) // 2

    def extract_corr(self, x: Tensor) -> Tensor:
        return self.biject_to_corr_cholesky(x[..., 2 * self.ndim:])

    def extract_scale_tril(self, x: Tensor) -> Tensor:
        return self.extract_scale(x).unsqueeze(-1) * self.extract_corr(x)

    def _get_dist(self, x: Tensor):
        return MultivariateNormal(loc=self.extract_loc(x), scale_tril=self.extract_scale_tril(x))
