from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Union, Mapping, Iterable, Generic, TYPE_CHECKING, Iterator, Any, Type

import pyro
from torch.utils.data import Dataset
from typing_extensions import TypeAlias
from torchdata.datapipes.iter import IterableWrapper


from ...utils.typing import _KT, _VT, _T, _Tin

_ConditionsT: TypeAlias = Union[Mapping[_KT, Iterable[_VT]], Iterable[Mapping[_KT, _VT]]]


class BaseConditionPipe(IterableWrapper, ABC, Generic[_T]):
    def __init__(self, iterable: Iterable[_T], conditions: _ConditionsT):
        super().__init__(iterable)
        self.conditions = conditions

    @staticmethod
    def transpose(mapping: Mapping[_KT, Iterable[_VT]]) -> Iterable[Mapping[_KT, _VT]]:
        return (dict(zip(mapping.keys(), v)) for v in zip(*mapping.values()))

    @property
    def _conditions(self):
        return self.transpose(self.conditions) if isinstance(self.conditions, Mapping) else self.conditions

    @abstractmethod
    def _conditioned_sample(self, condition: Mapping[_KT, _VT], _dataset: Iterator[_T]) -> _T: ...

    def __iter__(self):
        _dataset = super().__iter__()
        for condition in self._conditions:
            yield self._conditioned_sample(condition, _dataset)

    def __next__(self):
        raise NotImplementedError


class PyroConditionPipe(BaseConditionPipe[_T], Generic[_T]):
    conditions: Union[Mapping[Any, Iterable], Iterable[Mapping]]

    def _conditioned_sample(self, condition, _dataset):
        with pyro.condition(data=condition):
            return next(_dataset)


class BaseConditionableDataset(Dataset):
    @classmethod
    @property
    @abstractmethod
    def conditioner_cls(cls) -> Type[BaseConditionPipe]: ...
