from abc import abstractmethod, ABC
from functools import cached_property
from math import inf
from typing import Callable, ClassVar, Protocol, Union, Generic

from pyro.distributions import Normal, MultivariateNormal
from torch import Tensor
from torch.distributions import biject_to, TransformedDistribution, Distribution
from torch.distributions.constraints import Constraint, positive, corr_cholesky
from torch.nn import Module

from .nre import ParamPackerNRETail
from ...distributions.conundis.conundis_mixin import _constraintT, ConUnDisMixin
from ...sbi._typing import _KT
from ...utils.importing.attr import attr
from ...utils.nn.empty import _empty_module


class DistributionProtocol(Protocol):
    def log_prob(self, value: Tensor) -> Tensor: ...


class DistributionR(ABC):
    @abstractmethod
    def __call__(self, x: Tensor) -> Union[Distribution, DistributionProtocol]: ...


@attr.s
class NormalR(DistributionR):
    ndim: int

    biject_to_positive: ClassVar = biject_to(positive)

    @cached_property
    def input_size(self)  -> int:
        return 2 * self.ndim

    def extract_loc(self, x: Tensor) -> Tensor:
        return x[..., :self.ndim]

    def extract_scale(self, x: Tensor) -> Tensor:
        return self.biject_to_positive(x[..., self.ndim:2 * self.ndim])

    def __call__(self, x: Tensor):
        return Normal(self.extract_loc(x), self.extract_scale(x)).to_event(1)


class MVNR(NormalR):
    biject_to_corr_cholesky: ClassVar = biject_to(corr_cholesky)

    @cached_property
    def input_size(self) -> int:
        return 2 * self.ndim + (self.ndim * (self.ndim - 1)) // 2

    def extract_corr(self, x: Tensor) -> Tensor:
        return self.biject_to_corr_cholesky(x[..., 2 * self.ndim:])

    def extract_scale_tril(self, x: Tensor) -> Tensor:
        return self.extract_scale(x).unsqueeze(-1) * self.extract_corr(x)

    def __call__(self, x: Tensor):
        return MultivariateNormal(loc=self.extract_loc(x), scale_tril=self.extract_scale_tril(x))


@attr.s
class DistributionRWrapper(DistributionR, ABC):
    base_distr: DistributionR


@attr.s
class TransformerR(DistributionRWrapper):
    constraint: Constraint

    @cached_property
    def biject_to_constraint(self):
        return biject_to(self.constraint)

    def __call__(self, x: Tensor) -> Distribution:
        return TransformedDistribution(self.base_distr(x), [self.biject_to_constraint], validate_args=False)


@attr.s
class ConundisR(DistributionRWrapper):
    constraint_lower: _constraintT = -inf
    constraint_upper: _constraintT = inf

    def __call__(self, x: Tensor):
        return ConUnDisMixin.new_constrained(self.base_distr(x), constraint_lower=self.constraint_lower, constraint_upper=self.constraint_upper)


@attr.s
class DistRatioTail(ParamPackerNRETail[Tensor, _KT], Generic[_KT]):
    distr: Callable[[Tensor], DistributionProtocol]
    prior: DistributionProtocol
    xhead: Module = attr.ib(default=_empty_module)

    def _forward(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        return self.distr(self.xhead(x)).log_prob(theta) - self.prior.log_prob(theta)
