from __future__ import annotations

from functools import cached_property
from itertools import chain
from typing import Mapping, Union

import torch
from more_itertools import partition
from pyro.distributions import Delta, Normal
from pyro.distributions.constraints import lower_cholesky, positive
from pyro.nn import pyro_method, PyroParam, PyroSample
from torch import Tensor

from ..sampling_group import LocatedAndScaledSamplingGroupWithPrior
from ...utils.typing import AnyRegex, _Site


class PartialMultivariateNormalSamplingGroup(LocatedAndScaledSamplingGroupWithPrior):
    scale_full: Tensor
    scale_diag: Tensor
    scale_cross: Tensor
    zaux_full: Tensor
    zaux_diag: Tensor

    def __init__(self, sites, name='', diag=AnyRegex(),  # no match
                 init_scale_full: Union[torch.Tensor, float] = 1.,
                 init_scale_diag: Union[torch.Tensor, float] = 1.,
                 *args, **kwargs):
        self.init_scale_full = init_scale_full
        self.init_scale_diag = init_scale_diag

        self.diag_pattern = AnyRegex.get(diag)
        self.sites_full, self.sites_diag = (
            {site['name']: site for site in _}
            for _ in partition(lambda _: self.diag_pattern.match(_['name']), sites)
        )  # type: Mapping[str, _Site]

        # Here we ensure that full comes before diag
        super().__init__(chain(self.sites_full.values(), self.sites_diag.values()), name, *args, **kwargs)

        self.size_full, self.size_diag = (
            sum(self.sizes[site] for site in _)
            for _ in (self.sites_full, self.sites_diag)
        )  # type: int

    @cached_property
    def _jac(self):
        return self.jacobian(self.loc)

    @PyroParam(constraint=lower_cholesky, event_dim=2)
    def scale_full(self):
        return self._scale_matrix(self.init_scale_full, self._jac[:self.size_full])

    @PyroParam(constraint=positive, event_dim=1)
    def scale_diag(self):
        return self._scale_diagonal(self.init_scale_diag, self._jac[self.size_full:])

    @PyroParam(event_dim=2)
    def scale_cross(self):
        return self.loc.new_zeros(torch.Size((self.size_diag, self.size_full)))

    @cached_property
    def unit_normal(self):
        return Normal(self.loc.new_zeros(()), 1.)

    @PyroSample
    def zaux_full(self):
        return self.unit_normal.expand(torch.Size([self.size_full])).to_event(1)

    @PyroSample
    def zaux_diag(self):
        return self.unit_normal.expand(torch.Size([self.size_diag])).to_event(1)

    @property
    def half_log_det(self) -> Tensor:
        return self.scale_full.diagonal(dim1=-2, dim2=-1).log().sum(-1) + self.scale_diag.log().sum(-1)

    @property
    def _z_full(self) -> Tensor:
        return (self.scale_full @ self.zaux_full.unsqueeze(-1)).squeeze(-1)

    @property
    def _z_diag(self) -> Tensor:
        return (self.scale_cross @ self.zaux_full.unsqueeze(-1)).squeeze(-1) + self.scale_diag * self.zaux_diag

    def prior(self):
        return Delta(self.loc + torch.cat((self._z_full, self._z_diag), dim=-1), log_density=-self.half_log_det, event_dim=1)

    @pyro_method
    def sample_full(self):
        with self.grad_context:
            return self.unpack(
                self.loc[:self.size_full] + (self.scale_full @ self.zaux_full.unsqueeze(-1)).squeeze(-1),
                self.sites_full, guiding=False)
