import sys
from typing import Mapping, cast

from pyro.distributions.util import scale_and_mask
from pyro.poutine import is_validation_enabled
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.trace_struct import Trace
from pyro.util import warn_if_nan, warn_if_inf
from torch import Tensor

from .typing import _Site


class ClipppyTrace(Trace):
    nodes: Mapping[str, _Site]

    def as_valuedict(self) -> dict[str, Tensor]:
        return {key: site['value'] for key, site in self.nodes.items() if 'value' in site}

    def site_log_prob(self, name):
        site = self.nodes[name]

        # Copied from pyro.poutine.Trace
        if 'log_prob' not in site:
            try:
                log_p = site['fn'].log_prob(site["value"], *site["args"], **site["kwargs"])
            except ValueError as e:
                _, exc_value, traceback = sys.exc_info()
                shapes = self.format_shapes(last_site=site["name"])
                raise ValueError(
                    "Error while computing log_prob at site '{}':\n{}\n{}".format(name, exc_value, shapes)
                ).with_traceback(traceback) from e
            site["unscaled_log_prob"] = log_p
            log_p = scale_and_mask(log_p, site["scale"], site["mask"])
            site["log_prob"] = log_p
            site["log_prob_sum"] = log_p.sum()
            if is_validation_enabled():
                warn_if_nan(site["log_prob_sum"], "log_prob_sum at site '{}'".format(name))
                warn_if_inf(site["log_prob_sum"], "log_prob_sum at site '{}'".format(name), allow_neginf=True)

        return site['log_prob']

    def log_prob(self):
        self.compute_log_prob()
        return sum(site['log_prob'] for site in self.nodes.values())

    def stochastic_log_prob(self):
        return sum(map(self.site_log_prob, self.stochastic_nodes))

    def log_likelihood(self):
        return sum(map(self.site_log_prob, self.observation_nodes))

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
