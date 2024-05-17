from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from numbers import Number
from typing import Union, Mapping, Any, ContextManager, Iterable

from pyro.poutine import Trace
from torch import Tensor
from torch.utils.data import IterableDataset
from typing_extensions import TypeAlias

import clipppy
from . import _ValuesT, _WeightedT
from .conditionable import BaseConditionableDataset, PyroConditionPipe
from ...distributions.conundis import ConstrainingMessenger
from ...utils.messengers import CollectSitesMessenger
from ...utils.trace import ClipppyTrace
from ...utils.typing import _Distribution

_RangeBoundT: TypeAlias = Union[Number, Tensor, None]
_RangeT: TypeAlias = tuple[_RangeBoundT, _RangeBoundT]


@dataclass
class ClipppyDataset(BaseConditionableDataset, IterableDataset[_ValuesT]):
    conditioner_cls = PyroConditionPipe

    config: clipppy.Clipppy

    # TODO: smart setting batch_size on NREDataset
    batch_size: int = 0
    mock_args: Mapping[str, Any] = field(default_factory=lambda: dict(
        initting=False, conditioning=False, savename=False))

    @property
    def context(self) -> ContextManager:
        return nullcontext()

    def get_priors(self, param_names: Iterable[str]) -> Mapping[str, _Distribution]:
        with CollectSitesMessenger(*param_names) as trace:
            self.get_trace()
        return {name: site['fn'] for name, site in trace.items()}

    def get_prior_ranges(self, param_names: Iterable[str] = None, priors=None) -> Mapping[str, _RangeT]:
        return {key: (prior.support.lower_bound, prior.support.upper_bound)
                for key, prior in (priors or self.get_priors(param_names)).items()}

    def get_trace(self) -> ClipppyTrace:
        # TODO: retrying when an error occurs in simulator?!?
        # while True:
        #     try:
        with self.context:
            return self.config.mock(
                plate_stack=(self.batch_size,) if self.batch_size else None,
                **self.mock_args
            )
            # except ValueError:
            #     pass

    @staticmethod
    def get_values(trace: Trace) -> _ValuesT:
        return {key: val['value'] for key, val in trace.nodes.items() if 'value' in val}

    def __next__(self):
        return self.get_values(self.get_trace())

    def __iter__(self):
        return self


@dataclass
class CPDataset(ClipppyDataset):
    ranges: Mapping[str, _RangeT] = field(default_factory=dict)

    @property
    def context(self):
        return ConstrainingMessenger(self.ranges)

    @staticmethod
    def get_constrained_log_probs(trace: ClipppyTrace):
        return trace.compute_constrained_log_prob()


@dataclass
class CPWeightingPipe(IterableDataset[_WeightedT]):
    dataset: CPDataset

    def __iter__(self):
        return self

    def __next__(self) -> _WeightedT:
        trace = self.dataset.get_trace()
        return self.dataset.get_values(trace), self.dataset.get_constrained_log_probs(trace)
