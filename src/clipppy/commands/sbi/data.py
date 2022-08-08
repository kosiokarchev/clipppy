from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, cast, ContextManager, Generic, Iterable, Iterator, Mapping, Type, TYPE_CHECKING, Union

import pyro
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing_extensions import TypeAlias

from ... import clipppy
from ...distributions.conundis import ConstrainingMessenger
from ...utils import _KT, _T, _Tin, _Tout, _VT


__all__ = 'DoublePipe', 'SBIDataset', 'ClipppyDataset', 'CPDataset'

_OT: TypeAlias = Mapping[str, Tensor]
_OOT: TypeAlias = tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]


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

    def __next__(self):
        return self.split(next(self._dataset))


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

    def get_trace(self) -> pyro.poutine.Trace:
        while True:
            try:
                with self.context:
                    return self.config.mock(
                        plate_stack=(self.batch_size,) if self.batch_size else None,
                        **self.mock_args
                    )
            except ValueError:
                pass

    def __next__(self):
        return {key: val['value'] for key, val in self.get_trace().nodes.items()}

    def __iter__(self):
        return self


@dataclass
class CPDataset(ClipppyDataset):
    ranges: Mapping[str, tuple[Union[float, Tensor, None], Union[float, Tensor, None]]] = None

    @property
    def context(self):
        return ConstrainingMessenger(self.ranges)
