from dataclasses import dataclass, field
from functools import cached_property
from itertools import chain
from typing import Mapping, Iterable

import pyro
import torch
from more_itertools import always_iterable
from torch import Tensor

from ..clipppy import Clipppy
from ..distributions.conundis import ConstrainingMessenger
from ..distributions.histdist import HistDist
from ..sbi._typing import _MultiKT, _KT
from ..utils.typing import _Distribution
from ..utils.messengers import UpToMessenger


@dataclass
class ConstrainedInference:
    config: Clipppy
    samples: Mapping[_KT, Tensor]
    ranges: Mapping[_KT, tuple[Tensor, Tensor]]
    depmap: Mapping[_MultiKT, Iterable[_KT]]
    old_ranges: Mapping[_KT, tuple[Tensor, Tensor]] = field(default_factory=dict)

    def clp_trace(self, ranges):
        return self.config.mock(extra_messengers=(
            UpToMessenger(*chain(*map(always_iterable, self.depmap.keys())), *chain(*self.depmap.values())),
            ConstrainingMessenger(ranges=ranges),
            pyro.condition(data=self.samples)
        )).compute_constrained_log_prob()

    @cached_property
    def clps(self):
        clps_old = self.clp_trace(self.old_ranges)
        clps_new = self.clp_trace(self.ranges)

        return {
            group: (sum(clps_new[dep] for dep in deps) - sum(clps_old[dep] for dep in deps)).nan_to_num(nan=-float('inf'))
            for group, deps in self.depmap.items()
        }

    def cpriors(self, bins=100, **kwargs) -> Mapping[_MultiKT, _Distribution]:
        return {
            g: HistDist.from_samples(
                torch.stack([self.samples[p] for p in always_iterable(g)], dim=-1),
                bins=bins, **kwargs,
                weight=(self.clps[g] - self.clps[g].logsumexp(0)).exp()
            ) for g in self.depmap.keys()
        }
