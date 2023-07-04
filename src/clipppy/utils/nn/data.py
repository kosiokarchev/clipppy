from dataclasses import dataclass
from os import cpu_count
from typing import Mapping, TypeVar, Iterable, Generic

import torch
from more_itertools import one, unique_everseen
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .. import torch_get_default_device


def multiprocess_batch(dataset, batch_size, num_workers=None,
                       shuffle=True, pin_memory=True, drop_last=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=(num_workers if num_workers is not None else min(batch_size, cpu_count())),
        shuffle=shuffle, pin_memory=pin_memory, drop_last=drop_last,
        worker_init_fn=lambda *args, **kwargs: torch.set_default_tensor_type(torch.FloatTensor),
        generator=torch.Generator(device=torch_get_default_device()),
    )


_KT = TypeVar('_KT')


@dataclass
class TensorDataFrame(Dataset, Generic[_KT]):
    data: Mapping[_KT, Tensor]

    @property
    def device(self):
        return one(unique_everseen(t.device for t in self.data.values()))

    def __len__(self):
        return one(unique_everseen(map(len, self.data.values())))

    def __getitem__(self, item):
        return {key: val[item] for key, val in self.data.items()}

    def __iter__(self) -> Iterable[Mapping[_KT, Tensor]]:
        for val in zip(*self.data.values()):
            yield dict(zip(self.data.keys(), val))

    @dataclass
    class BatchedIterator:
        df: 'TensorDataFrame'
        batch_size: int
        shuffle: bool = True

        def __iter__(self):
            for i in (torch.randperm if self.shuffle else torch.arange)(len(self.df), device=self.df.device, dtype=int).split(self.batch_size):
                yield self.df[i]

    def batched(self, batch_size, shuffle=True):
        return self.BatchedIterator(self, batch_size, shuffle)
