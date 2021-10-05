from __future__ import annotations

from typing import Mapping

import torch
from more_itertools import all_equal
from pyro.nn import PyroParam
from torch import Size

from .pmvn import PartialMultivariateNormalSamplingGroup


class HierarchicPartialMultivariateNormalSamplingGroup(PartialMultivariateNormalSamplingGroup):
    def __init__(self, *args, hdims: Mapping[str, int], **kwargs):
        super().__init__(*args, **kwargs)

        hdims = {name: len(self.shapes[name]) - nevent for name, nevent in hdims.items()}
        self.corr_names = list(hdims.keys())
        assert all(name in self.sites_diag for name in self.corr_names)

        self.corr_batch = {name: self.shapes[name][:nevent] for name, nevent in hdims.items()}
        assert all_equal(self.corr_batch.values())
        self.corr_batch: Size = next(iter(self.corr_batch.values()))

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

    @property
    def corr_matrix(self):
        ret = self.loc.new_zeros(self.corr_batch + 2*(self.corr_size-1,))
        return ret.flatten(-2).scatter_(-1, self.corr_tril_indices, self.corr).unflatten(-1, ret.shape[-2:])

    @property
    def _z_diag(self):
        ret = super()._z_diag
        ret[..., self.corr_target_indices[..., 1:]] += (self.corr_matrix @ self.zaux_diag[..., self.corr_target_indices[..., :-1]].unsqueeze(-1)).squeeze(-1)
        return ret
