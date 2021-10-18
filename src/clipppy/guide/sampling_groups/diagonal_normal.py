from __future__ import annotations

from pyro.distributions import Normal
from pyro.distributions.constraints import positive
from pyro.nn import PyroParam
from torch import Tensor

from ..sampling_group import LocatedAndScaledSamplingGroupWithPrior


class DiagonalNormalSamplingGroup(LocatedAndScaledSamplingGroupWithPrior):
    scale: Tensor

    @PyroParam(event_dim=1, constraint=positive)
    def scale(self):
        return self._scale_diagonal(self.init_scale, self.jacobian(self.loc))

    def prior(self):
        return Normal(self.loc, self.scale).to_event(1)
