from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Generic, TYPE_CHECKING, TypeVar, Callable, Union

import attr
import pyro.distributions
import torch.autograd
from pyro.distributions import MultivariateNormal, Normal, TransformedDistribution
from torch import Tensor
from torch.distributions import biject_to
from torch.distributions.constraints import Constraint, corr_cholesky, positive, real
from torch.nn import LazyLinear

from . import _HeadOoutT, BaseSBITail, ParamPackerSBITail
from ...utils.nn import AttrsModule, _KT

if TYPE_CHECKING:
    from typing import Type

    # because TransformedDistribution cannot be found in pyro.distributions
    TransformedDistribution: Union[Type[torch.distributions.TransformedDistribution], Type[pyro.distributions.Distribution]]


_DistributionT = TypeVar('_DistributionT', bound=Union[pyro.distributions.Distribution, torch.distributions.Distribution])


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


class BaseNPETail(BaseSBITail[_HeadOoutT, NPEResult[_DistributionT], _KT], Generic[_HeadOoutT, _DistributionT, _KT], ABC):
    pass


class NPETail(ParamPackerSBITail[_HeadOoutT, _DistributionT, _KT], BaseNPETail[_HeadOoutT, _DistributionT, _KT], Generic[_HeadOoutT, _DistributionT, _KT], ABC):
    @abstractmethod
    def _get_dist(self, x: _HeadOoutT) -> _DistributionT: ...

    def get_dist(self, x: _HeadOoutT) -> _DistributionT:
        return self._get_dist(x)

    def _forward(self, theta: Tensor, x: _HeadOoutT, **kwargs) -> NPEResult[_DistributionT]:
        return NPEResult(theta, self._get_dist(x), **kwargs)


@attr.s
class ConstrainedNPETail(NPETail[_HeadOoutT, TransformedDistribution, _KT], Generic[_HeadOoutT, _KT], ABC):
    constraint: Constraint = attr.ib(default=real, kw_only=True)

    @cached_property
    def biject_to_constraint(self):
        return biject_to(self.constraint)

    def get_dist(self, x: _HeadOoutT) -> _DistributionT:
        return TransformedDistribution(super()._get_dist(x), [self.biject_to_constraint])


@attr.s
class ParametrizedNPETail(NPETail[Tensor, _DistributionT, _KT], AttrsModule, Generic[_DistributionT, _KT]):
    net: Callable[[Tensor], Tensor] = attr.ib(default=None, kw_only=True)

    def __attrs_post_init__(self):
        if self.net is None:
            self.net = LazyLinear(self.event_size)

    @property
    @abstractmethod
    def event_size(self) -> int: ...

    def get_dist(self, x: _HeadOoutT) -> _DistributionT:
        return super()._get_dist(self.net(x))


@attr.s
class NormalTail(ParametrizedNPETail[TransformedDistribution, _KT], ConstrainedNPETail[Tensor, _KT], Generic[_KT]):
    ndim: int
    _biject_to_positive: ClassVar = biject_to(positive)

    @cached_property
    def event_size(self):
        return 2 * self.ndim

    def extract_loc(self, x: Tensor) -> Tensor:
        return x[..., :self.ndim]

    def extract_scale(self, x: Tensor) -> Tensor:
        return self._biject_to_positive(x[..., self.ndim:2 * self.ndim])

    def _get_dist(self, x: Tensor):
        return Normal(self.extract_loc(x), self.extract_scale(x))


class MVNTail(NormalTail[_KT], Generic[_KT]):
    _biject_to_corr_cholesky: ClassVar = biject_to(corr_cholesky)

    @cached_property
    def event_size(self):
        return 2 * self.ndim + (self.ndim * (self.ndim - 1)) // 2

    def extract_corr(self, x: Tensor) -> Tensor:
        return self._biject_to_corr_cholesky(x[..., 2 * self.ndim:])

    def extract_scale_tril(self, x: Tensor) -> Tensor:
        return self.extract_scale(x).unsqueeze(-1) * self.extract_corr(x)

    def _get_dist(self, x: Tensor):
        return MultivariateNormal(loc=self.extract_loc(x), scale_tril=self.extract_scale_tril(x))
