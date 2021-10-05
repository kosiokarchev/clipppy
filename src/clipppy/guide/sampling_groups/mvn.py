from __future__ import annotations

import torch
from pyro.distributions import MultivariateNormal
from pyro.distributions.constraints import corr_cholesky, positive
from pyro.nn import PyroParam
from torch import Tensor

from ..sampling_group import LocatedSamplingGroupWithPrior, ScaledSamplingGroup


class MultivariateNormalSamplingGroup(ScaledSamplingGroup, LocatedSamplingGroupWithPrior):
    scale_tril: Tensor

    @PyroParam(constraint=positive, event_dim=1)
    def scale(self):
        return self._scale_diagonal(self.init_scale, self.jacobian(self.loc))

    @PyroParam(constraint=corr_cholesky, event_dim=2)
    def corr_cholesky(self):
        return torch.eye(len(self.loc), device=self.loc.device, dtype=self.loc.dtype)

    def prior(self):
        return MultivariateNormal(self.loc, scale_tril=self.scale.unsqueeze(-1) * self.corr_cholesky)
