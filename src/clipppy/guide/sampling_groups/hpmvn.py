from __future__ import annotations

from typing import Mapping

import torch
from more_itertools import all_equal
from pyro.distributions.constraints import lower_cholesky
from pyro.nn import PyroParam
from torch import Size

from .pmvn import PartialMultivariateNormalSamplingGroup


class HierarchicPartialMultivariateNormalSamplingGroup(PartialMultivariateNormalSamplingGroup):
    def __init__(self, *args, hdims: Mapping[str, int], init_scale_const_corr=1., **kwargs):
        super().__init__(*args, **kwargs)

        hdims = {name: len(self.shapes[name]) - nevent for name, nevent in hdims.items()}
        self.corr_names = list(hdims.keys())
        assert all(name in self.sites_diag for name in self.corr_names)

        self.corr_batch = {name: self.shapes[name][:nevent] for name, nevent in hdims.items()}
        assert all_equal(self.corr_batch.values())
        self.corr_batch: Size = next(iter(self.corr_batch.values()))
        assert len(self.corr_batch) > 0

        self.corr_sizes = {name: self.shapes[name][nevent:].numel() for name, nevent in hdims.items()}
        self.corr_size: int = sum(self.corr_sizes.values())
        self.corr_poss = {name: s for snext in [0] for name in self.corr_names for s in [snext] for snext in [snext+self.corr_sizes[name]]}

        self.corr = PyroParam(self.loc.new_zeros(self.corr_batch + (self.corr_size*(self.corr_size-1) // 2,)),
                              event_dim=len(self.corr_batch) + 1)
        self.corr_tril_indices = torch.tril_indices(*2 * (self.corr_size-1,), device=self.loc.device)
        self.corr_tril_indices = self.corr_tril_indices.expand(self.corr_batch + self.corr_tril_indices.shape).movedim(-2, 0)
        self.corr_tril_indices = (self.corr_size-1) * self.corr_tril_indices[0] + self.corr_tril_indices[1]
        self.corr_target_indices = torch.cat(tuple(
            self.poss[name]-self.size_full + torch.arange(self.sizes[name], device=self.loc.device).reshape(*self.shapes[name][:hdims[name]], -1)
            for name in self.corr_names
        ), -1)

        self.init_scale_const_corr = init_scale_const_corr

    @PyroParam(constraint=lower_cholesky, event_dim=2)
    def const_corr(self):
        # TODO: simplify const_corr_scale jacobian?
        return self._scale_diagonal(
            self.init_scale_const_corr, torch.cat(tuple(
                torch.atleast_1d(tr.log_abs_det_jacobian(z, tr(z)).exp()).flatten()
                for name in self.corr_names
                for z in [self.unpack_site(self.loc, name).reshape(self.shapes[name]).flatten(0, len(self.corr_batch)-1).mean(0)]
                for tr in [self.transforms[name]]
            ), -1)
        ) * torch.eye(self.corr_size, device=self.loc.device, dtype=self.loc.dtype)

    @property
    def corr_matrix(self):
        ret = self.loc.new_zeros(self.corr_batch + 2*(self.corr_size-1,))
        return ret.flatten(-2).scatter_(-1, self.corr_tril_indices, self.corr).unflatten(-1, ret.shape[-2:])

    @property
    def zaux_corr(self):
        return self.zaux_diag[..., self.corr_target_indices]

    @property
    def _z_diag(self):
        ret = super()._z_diag
        zaux = self.zaux_corr
        ret[..., self.corr_target_indices[..., 1:]] += (self.corr_matrix @ zaux[..., :-1].unsqueeze(-1)).squeeze(-1)
        ret[..., self.corr_target_indices] += (self.const_corr @ zaux.unsqueeze(-1)).squeeze(-1).sum(tuple(range(-1-len(self.corr_batch), -1)), keepdims=True) / self.corr_batch.numel()**0.5
        return ret
