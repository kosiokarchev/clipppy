from __future__ import annotations

from functools import singledispatch

from pyro.distributions import Independent, ExpandedDistribution
from torch import Tensor
from torch.distributions.utils import _sum_rightmost

from .extra_dimensions import LeftIndependent


@singledispatch
def process_log_prob(w, p: Tensor) -> Tensor:
    # Long live functional programming! :(
    raise NotImplementedError(f'type {type(w)} not supported')


@process_log_prob.register
def process_log_prob_event(w: Independent, p: Tensor) -> Tensor:
    return _sum_rightmost(p, w.reinterpreted_batch_ndims)


@process_log_prob.register
def process_log_prob_batch(w: ExpandedDistribution, p: Tensor) -> Tensor:
    return p.expand(w.batch_shape)


process_log_prob.register(LeftIndependent)(LeftIndependent.process_log_prob)
