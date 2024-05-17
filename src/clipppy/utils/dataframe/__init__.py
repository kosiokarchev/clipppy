from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import singledispatchmethod
from math import ceil
from typing import Mapping, Iterable

import torch
from more_itertools import one, unique_everseen
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import TypeAlias

from ..typing import _Tensor_like


_KT: TypeAlias = str


class AbstractTensorDataFrame(Dataset[Mapping[_KT, Tensor]], ABC):
    device = None

    @dataclass
    class BatchedIterator:
        df: 'AbstractTensorDataFrame'
        batch_size: int
        shuffle: bool = True

        seed: int = 0
        _epoch = 0

        def __len__(self):
            return int(ceil(len(self.df) / self.batch_size))

        def _idx(self):
            if self.shuffle:
                return torch.randperm(len(self.df), device=self.df.device, generator=torch.Generator().manual_seed(self.seed + self._epoch))
            else:
                return torch.arange(len(self.df), device=self.df.device, dtype=torch.long)

        def __iter__(self):
            self._epoch += 1
            for i in self._idx().split(self.batch_size):
                yield self.df[i]

    def batched(self, batch_size, shuffle=True):
        return self.BatchedIterator(self, batch_size, shuffle)

    @abstractmethod
    def __len__(self): ...

    @abstractmethod
    def _getitem(self, item) -> Mapping[_KT, Tensor]:  ...

    @abstractmethod
    def _getitem_column(self, item: str) -> _Tensor_like: ...

    @singledispatchmethod
    def __getitem__(self, item) -> Mapping[_KT, Tensor]:
        return self._getitem(item)

    @__getitem__.register
    def _(self, item: str) -> _Tensor_like:
        return self._getitem_column(item)

    def __getitems__(self, item: list[str]) -> Mapping[_KT, _Tensor_like]:
        return (
            {key: self[key] for key in item}
            if all(isinstance(_, str) for _ in item)
            else self._getitem(item)
        )

    __getitem__.register(list)(__getitems__)


@dataclass
class TensorDataFrame(AbstractTensorDataFrame):
    data: Mapping[_KT, _Tensor_like] = field(repr=False)

    @property
    def device(self):
        return one(unique_everseen(t.device for t in self.data.values() if torch.is_tensor(t)))

    def __len__(self):
        return one(unique_everseen(map(len, self.data.values())))

    def _getitem(self, item) -> Mapping[_KT, Tensor]:
        return {key: val[item] for key, val in self.data.items()}

    def _getitem_column(self, item: str) -> _Tensor_like:
        return self.data[item]

    def __iter__(self) -> Iterable[Mapping[_KT, Tensor]]:
        for val in zip(*self.data.values()):
            yield dict(zip(self.data.keys(), val))
