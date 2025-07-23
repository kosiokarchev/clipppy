from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import ClassVar, Generic, TYPE_CHECKING, TypeVar, Callable, Mapping

import attr
import pyro.distributions
import torch.autograd
from more_itertools import always_iterable
from pyro.distributions import MultivariateNormal, Normal, TransformedDistribution, ConditionalTransformedDistribution, \
    ConditionalTransform
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.distributions.transforms import conditional_spline_autoregressive, permute
from torch import Tensor, Size
from torch.distributions import biject_to
from torch.distributions.constraints import Constraint, corr_cholesky, positive, real, interval, independent
from torch.nn import LazyLinear, Sequential, Module

from . import _HeadOoutT, BaseSBITail, ParamPackerSBITail
from .._typing import _KT, _MultiKT
from ...utils.nn import AttrsModule, extract_extra_buffers
from ...utils.nn.empty import _empty_module

if TYPE_CHECKING:
    from typing import Type

    # because TransformedDistribution cannot be found in pyro.distributions
    TransformedDistribution: Type[torch.distributions.TransformedDistribution] | Type[pyro.distributions.Distribution]


_DistributionT = TypeVar('_DistributionT', bound=pyro.distributions.Distribution | torch.distributions.Distribution)


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
        return NPEResult(theta, self.get_dist(x), **kwargs)


@attr.s(eq=False, auto_attribs=True)
class ConstrainedNPETail(NPETail[_HeadOoutT, TransformedDistribution, _KT], Generic[_HeadOoutT, _KT], ABC):
    constraint: Constraint = attr.ib(default=real, kw_only=True)

    @classmethod
    def constraint_from_samples(cls, samples: Mapping[_KT, Tensor], names: _MultiKT, margin=0.01):
        mins, maxs = map(torch.stack, zip(*map(torch.aminmax, (samples[key] for key in always_iterable(names)))))
        marg = margin * (maxs - mins)
        return independent(interval(mins-marg, maxs+marg), 1)

    @cached_property
    def biject_to_constraint(self):
        return biject_to(self.constraint)

    def _apply(self, fn, recurse=True):
        # TODO: register transform buffers (hacked)
        # for b in extract_buffers(self.biject_to_constraint, Transform):
        for b in extract_extra_buffers(self):
            if b._use_count() <= 2:
                torch.utils.swap_tensors(b, fn(b))

        return super()._apply(fn, recurse)

    def get_dist(self, x: _HeadOoutT) -> _DistributionT:
        return TransformedDistribution(super().get_dist(x), [self.biject_to_constraint])


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class ParametrizedNPETail(NPETail[_HeadOoutT, _DistributionT, _KT], AttrsModule, Generic[_HeadOoutT, _DistributionT, _KT]):
    net: Module | Callable[[_HeadOoutT], Tensor] = _empty_module
    add_last: bool = True

    def __attrs_post_init__(self):
        if self.add_last:
            self.net = Sequential(self.net, LazyLinear(self.event_size))

    @property
    @abstractmethod
    def event_size(self) -> int: ...

    def get_dist(self, x: _HeadOoutT) -> _DistributionT:
        return super().get_dist(self.net(x))


@attr.s(eq=False, auto_attribs=True)
class NormalTail(ParametrizedNPETail[_HeadOoutT, TransformedDistribution, _KT], ConstrainedNPETail[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
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


class MVNTail(NormalTail[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
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


@attr.s(eq=False, auto_attribs=True)
class NFTail(ParametrizedNPETail[_HeadOoutT, ConditionalTransformedDistribution, _KT], ConstrainedNPETail[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    ndim: int
    _event_size: int
    transform: ConditionalTransform

    def event_size(self) -> int:
        return self._event_size

    @classmethod
    def spline_autoregressive(cls, ndim: int, context_size: int, count_bins: int = 16, nlayers: int = 5, hidden_size: int | list[int] = None, nhidden: int = 2, bound: float = 5., **kwargs):
        if hidden_size is None:
            hidden_size = max(ndim * count_bins, context_size)
        if isinstance(hidden_size, int):
            hidden_size = nhidden * [hidden_size]

        return cls(ndim, context_size, ConditionalComposeTransformModule(list(chain.from_iterable(
            (conditional_spline_autoregressive(ndim, context_size, hidden_dims=hidden_size, count_bins=count_bins, bound=bound),
             permute(ndim))
            for _ in range(nlayers)
        ))), **kwargs)

    def __attrs_post_init__(self):
        self._cdist = ConditionalTransformedDistribution(
            Normal(0., 1.).expand(Size((self.ndim,))).to_event(1),
            transforms=[self.transform]
        )

    def _get_dist(self, x: Tensor):
        return self._cdist.condition(x)
