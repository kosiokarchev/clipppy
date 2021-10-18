from __future__ import annotations

import torch
from pyro.distributions import MultivariateNormal
from pyro.distributions.constraints import corr_cholesky, positive
from pyro.nn import PyroParam
from torch import Tensor

from ..sampling_group import LocatedAndScaledSamplingGroupWithPrior


class MultivariateNormalSamplingGroup(LocatedAndScaledSamplingGroupWithPrior):
    scale_tril: Tensor

    @PyroParam(constraint=positive, event_dim=1)
    def scale(self):
        return self._scale_diagonal(self.init_scale, self.jacobian(self.loc))

    @PyroParam(constraint=corr_cholesky, event_dim=2)
    def corr_cholesky(self):
        return torch.eye(len(self.loc), device=self.loc.device, dtype=self.loc.dtype)

    @property
    def scale_tril(self):
        return self.scale.unsqueeze(-1) * self.corr_cholesky

    def prior(self):
        return MultivariateNormal(self.loc, scale_tril=self.scale_tril)
