from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Generic, Iterable, Iterator, Mapping, cast

from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from typing_extensions import TypeAlias

from .conditionable import PyroConditionPipe, BaseConditionableDataset
from .._typing import SBIBatch
from ...utils.typing import _Tin


_ValuesT: TypeAlias = Mapping[str, Tensor]
_WeightsT: TypeAlias = Mapping[str, Tensor]
_WeightedT: TypeAlias = tuple[_ValuesT, _WeightsT]
_SBIBatchT: TypeAlias = SBIBatch[str]


@dataclass(kw_only=True)
class SBIProcessor:
    param_names: Iterable[str]
    obs_names: Iterable[str]

    def split(self, values: _ValuesT) -> tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]:
        return cast(tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]], tuple(
            OrderedDict((name, values[name]) for name in names)
            for names in (self.param_names, self.obs_names)
        ))


@dataclass
class AbstractSBIDataset(SBIProcessor, IterableDataset, Generic[_Tin], ABC):
    iterable: Iterable[_Tin]

    @abstractmethod
    def process(self, item: _Tin) -> _SBIBatchT: ...

    def __iter__(self) -> Iterator[_SBIBatchT]:
        for values in self.iterable:
            yield self.process(values)


@dataclass
class SBIDataset(AbstractSBIDataset[_ValuesT]):
    def process(self, item: _ValuesT) -> _SBIBatchT:
        return SBIBatch(*self.split(item))


@dataclass
class WeightedSBIDataset(AbstractSBIDataset[_WeightedT]):
    # TODO: unsqueezing weights
    param_event_dims: Mapping = field(default_factory=dict)

    def process(self, item: _WeightedT) -> _SBIBatchT:
        values, weights = item
        return SBIBatch(*self.split(values), {
            key: val.reshape(val.shape + self.param_event_dims.get(key, 0)*(1,))
            for key, val in weights.items()
        })


class SBIDataLoader(DataLoader):
    dataset: SBIDataset
