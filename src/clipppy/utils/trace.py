from typing import Mapping, cast

from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.trace_struct import Trace
from torch import Tensor

from .typing import _Site


class ClipppyTrace(Trace):
    nodes: Mapping[str, _Site]

    def log_prob(self):
        self.compute_log_prob()
        return sum(site['log_prob'] for site in self.nodes.values())

    def compute_constrained_log_prob(self) -> Mapping[str, Tensor]:
        from ..distributions.constrained import constrained_log_prob

        ret = {}
        for site in self.nodes.values():
            if 'constrained_log_prob' not in site:
                site['constrained_log_prob'] = constrained_log_prob(site['fn'])
            ret[site['name']] = site['constrained_log_prob']
        return ret

    def constrained_log_prob(self) -> Tensor:
        return sum(self.compute_constrained_log_prob().values())


class ClipppyTraceMessenger(TraceMessenger):
    def get_trace(self) -> ClipppyTrace:
        res = super().get_trace()
        res.__class__ = ClipppyTrace
        return cast(ClipppyTrace, res)
