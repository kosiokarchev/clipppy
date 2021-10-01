from __future__ import annotations

from functools import cached_property
from itertools import chain
from typing import cast, Mapping, Union

import torch
from more_itertools import partition
from pyro import distributions as dist
from pyro.distributions import constraints
from pyro.nn import pyro_method, PyroParam, PyroSample

from ..guide.sampling_group import LocatedSamplingGroupWithPrior, SamplingGroup
from ..utils.typing import _Site, AnyRegex


# This should be the same as EasyGuide's map_estimate,
# but slower because of (needless?) transformations, unpacking, etc.
# TODO: simplify DeltaSamplingGroup?
class DeltaSamplingGroup(SamplingGroup):
    def __init__(self, sites, name='', *args, **kwargs):
        super().__init__(sites, name)

        self.loc = PyroParam(self.init[self.mask], event_dim=1)

    # Emulate EasyGuide's map_estimate implementation
    include_det_jac = False

    def _sample(self, infer=None) -> torch.Tensor:
        return cast(torch.Tensor, self.loc)


class DiagonalNormalSamplingGroup(LocatedSamplingGroupWithPrior):
    def __init__(self, sites, name='', init_scale: Union[torch.Tensor, float] = 1., *args, **kwargs):
        super().__init__(sites, name, *args, **kwargs)

        self.scale = PyroParam(self._scale_diagonal(init_scale, self.jacobian(self.loc)),
                               event_dim=1, constraint=constraints.positive)

    def prior(self):
        return dist.Normal(self.loc, self.scale).to_event(1)


class MultivariateNormalSamplingGroup(LocatedSamplingGroupWithPrior):
    def __init__(self, sites, name='', init_scale: Union[torch.Tensor, float] = 1., *args, **kwargs):
        super().__init__(sites, name, *args, **kwargs)

        self.scale_tril = PyroParam(self._scale_matrix(init_scale, self.jacobian(self.loc)),
                                    event_dim=2, constraint=constraints.lower_cholesky)

    def prior(self):
        return dist.MultivariateNormal(self.loc, scale_tril=self.scale_tril)


class PartialMultivariateNormalSamplingGroup(LocatedSamplingGroupWithPrior):
    def __init__(self, sites, name='', diag=AnyRegex(),  # no match
                 init_scale_full: Union[torch.Tensor, float] = 1.,
                 init_scale_diag: Union[torch.Tensor, float] = 1.,
                 *args, **kwargs):
        self.diag_pattern = AnyRegex.get(diag)
        self.sites_full, self.sites_diag = (
            {site['name']: site for site in _}
            for _ in partition(lambda _: self.diag_pattern.match(_['name']), sites)
        )  # type: Mapping[str, _Site]

        super().__init__(chain(self.sites_full.values(), self.sites_diag.values()), name, *args, **kwargs)

        self.size_full, self.size_diag = (
            sum(self.sizes[site] for site in _)
            for _ in (self.sites_full, self.sites_diag)
        )  # type: int

        jac = self.jacobian(self.loc)

        self.scale_full = PyroParam(self._scale_matrix(init_scale_full, jac[:self.size_full]),
                                    event_dim=2, constraint=constraints.lower_cholesky)
        self.scale_cross = PyroParam(self.loc.new_zeros(torch.Size((self.size_diag, self.size_full))), event_dim=2)
        self.scale_diag = PyroParam(self._scale_diagonal(init_scale_diag, jac[self.size_full:]),
                                    event_dim=1, constraint=constraints.positive)

    @cached_property
    def unit_normal(self):
        return dist.Normal(self.loc.new_zeros(()), 1.)

    @PyroSample
    def guide_z_aux_full(self):
        return self.unit_normal.expand(torch.Size([self.size_full])).to_event(1)

    @PyroSample
    def guide_z_aux_diag(self):
        return self.unit_normal.expand(torch.Size([self.size_diag])).to_event(1)

    @pyro_method
    def sample_full(self):
        with self.grad_context:
            return self.unpack(
                self.loc[:self.size_full] + (self.scale_full @ self.guide_z_aux_full.unsqueeze(-1)).squeeze(-1),
                self.sites_full, guiding=False)

    @property
    def half_log_det(self):
        return self.scale_full.diagonal(dim1=-2, dim2=-1).log().sum(-1) + self.scale_diag.log().sum(-1)

    def prior(self):
        zfull = self.guide_z_aux_full.unsqueeze(-1)
        zdiag = self.guide_z_aux_diag
        return dist.Delta(self.loc + torch.cat((
            (self.scale_full @ zfull).squeeze(-1),
            (self.scale_cross @ zfull).squeeze(-1) + self.scale_diag * zdiag
        ), dim=-1), log_density=-self.half_log_det, event_dim=1)
