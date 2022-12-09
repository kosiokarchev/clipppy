from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from numbers import Number
from typing import (
    Any, cast, Collection, ContextManager, Generic, Iterable, Iterator,
    Mapping, Type, TYPE_CHECKING, Union)

import pyro
import torch
from more_itertools import consume
from pyro.poutine import Trace
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing_extensions import TypeAlias

from .. import clipppy
from ..distributions.conundis import ConstrainingMessenger
from ..utils import _KT, _T, _Tin, _Tout, _VT
from ..utils.messengers import CollectSitesMessenger, RequiresGradMessenger
from ..utils.typing import _Distribution


_OT: TypeAlias = Mapping[str, Tensor]
_OOT: TypeAlias = tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]

_RangeBoundT: TypeAlias = Union[Number, Tensor, None]
_RangeT: TypeAlias = tuple[_RangeBoundT, _RangeBoundT]


@dataclass
class DataPipe(IterableDataset[_Tout], ABC, Generic[_Tin, _Tout]):
    dataset: Iterable[_Tin]
    _dataset: Iterator[_Tin] = field(init=False, repr=False)

    def __iter__(self) -> Iterator[_Tout]:
        self._dataset = iter(self.dataset)
        return self

    @abstractmethod
    def __next__(self) -> _Tout: ...


_ConditionsT: TypeAlias = Union[Mapping[_KT, Iterable[_VT]], Iterable[Mapping[_KT, _VT]]]


@dataclass
class BaseConditionPipe(DataPipe[_T, _T], ABC, Generic[_T]):
    conditions: _ConditionsT

    # TODO: bug in PyCharm that doesn't recognise the __init__ of dataclass
    #   when inspecting a variable annotated with Type[SomeDataclass] ...
    if TYPE_CHECKING:
        def __init__(self, dataset: Iterable[_Tin], conditions: _ConditionsT): ...

    @staticmethod
    def transpose(mapping: Mapping[_KT, Iterable[_VT]]) -> Iterable[Mapping[_KT, _VT]]:
        return (dict(zip(mapping.keys(), v)) for v in zip(*mapping.values()))

    @property
    def _conditions(self):
        return self.transpose(self.conditions) if isinstance(self.conditions, Mapping) else self.conditions

    @abstractmethod
    def _conditioned_sample(self, condition: Mapping[_KT, _VT], _dataset: Iterator[_T]) -> _T: ...

    def __iter__(self):
        _dataset = iter(self.dataset)
        for condition in self._conditions:
            yield self._conditioned_sample(condition, _dataset)

    def __next__(self):
        raise NotImplementedError


class PyroConditionPipe(BaseConditionPipe[_T], Generic[_T]):
    conditions: Union[Mapping[Any, Iterable], Iterable[Mapping]]

    def _conditioned_sample(self, condition, _dataset):
        with pyro.condition(data=condition):
            return next(_dataset)


class DoublePipe(DataPipe[_Tin, tuple[_Tin, _Tin]], Generic[_Tin]):
    def __next__(self):
        return next(self._dataset), next(self._dataset)


@dataclass
class SBIDataset(DataPipe[_OT, _OOT]):
    dataset: Union[Iterable[_OT], BaseConditionableDataset[_OT]]

    param_names: Iterable[str]
    obs_names: Iterable[str]

    def split(self, values: _OT) -> _OOT:
        return cast(_OOT, tuple(
            OrderedDict((name, values[name]) for name in names)
            for names in (self.param_names, self.obs_names)
        ))

    def __next__(self) -> _OOT:
        return self.split(next(self._dataset))


class GASBIDataset(SBIDataset):
    param_names: Collection[str]

    def __next__(self):
        with RequiresGradMessenger(self.param_names, func_other=Tensor.detach):
            trace: Trace = pyro.poutine.trace(super().__next__).get_trace()
        log_prob = sum(val['fn'].log_prob(val['value'])
                       for val in trace.nodes.values() if 'fn' in val)
        log_prob.backward(torch.ones_like(log_prob))
        # return trace.nodes['_RETURN']['value']
        params, obs = trace.nodes['_RETURN']['value']
        consume(map(partial(Tensor.requires_grad_, requires_grad=False),
                    chain(params.values(), obs.values())))
        return params, obs


class SBIDataLoader(DataLoader):
    dataset: SBIDataset


class BaseConditionableDataset(Dataset):
    @classmethod
    @property
    @abstractmethod
    def conditioner_cls(cls) -> Type[BaseConditionPipe]: ...


@dataclass
class ClipppyDataset(BaseConditionableDataset, IterableDataset[_OT]):
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

    def get_trace(self) -> Trace:
        # TODO: retrying when an error occurs in simulator?!?
        while True:
            try:
                with self.context:
                    return self.config.mock(
                        plate_stack=(self.batch_size,) if self.batch_size else None,
                        **self.mock_args
                    )
            except ValueError:
                pass

    @staticmethod
    def get_values(trace: Trace) -> _OT:
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
