from __future__ import annotations

from pyro.distributions import Delta

from ..sampling_group import LocatedSamplingGroupWithPrior


# This should be the same as EasyGuide's map_estimate,
# but slower because of (needless?) transformations, unpacking, etc.
# TODO: simplify DeltaSamplingGroup?
class DeltaSamplingGroup(LocatedSamplingGroupWithPrior):
    # Emulate EasyGuide's map_estimate implementation
    include_det_jac = False

    def prior(self):
        return Delta(self.loc).to_event(1)
