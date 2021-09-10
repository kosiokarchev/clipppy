from __future__ import annotations

import re
from itertools import chain
from typing import cast, Mapping, Union

import torch
from more_itertools import partition
from pyro import distributions as dist
from pyro.distributions import constraints
from pyro.nn import PyroParam, PyroSample

from ..guide.sampling_group import LocatedSamplingGroupWithPrior, SamplingGroup
from ..utils import _nomatch
from ..utils.typing import _Site


# This should be the same as EasyGuide's map_estimate,
# but slower because of (needless?) transformations, unpacking, etc.
# TODO: simplify DeltaSamplingGroup?
class DeltaSamplingGroup(SamplingGroup):
    def __init__(self, sites, name='', *args, **kwargs):
        super().__init__(sites, name)

        self.loc = PyroParam(self.init[self.mask], event_dim=1)

    # Emulate EasyGuide's map_estimate implementation
    include_det_jac = False

    def _sample(self, infer) -> torch.Tensor:
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
    def __init__(self, sites, name='', diag=_nomatch,
                 init_scale_full: Union[torch.Tensor, float] = 1.,
                 init_scale_diag: Union[torch.Tensor, float] = 1.,
                 *args, **kwargs):
        self.diag_pattern = re.compile(diag)
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

        self.guide_z_aux = PyroSample(dist.Normal(self.loc.new_zeros(()), 1.).expand(self.event_shape).to_event(1))

    @property
    def half_log_det(self):
        return self.scale_full.diagonal(dim1=-2, dim2=-1).log().sum(-1) + self.scale_diag.log().sum(-1)

    def prior(self):
        z_aux = self.guide_z_aux
        zfull, zdiag = z_aux[..., :self.size_full, None], z_aux[..., self.size_full:]
        return dist.Delta(self.loc + torch.cat((
            (self.scale_full @ zfull).squeeze(-1),
            (self.scale_cross @ zfull).squeeze(-1) + self.scale_diag * zdiag
        ), dim=-1), log_density=-self.half_log_det, event_dim=1)
