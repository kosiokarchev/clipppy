from typing import Iterator

import torch

import phytorchx
from ..utils.typing import _T
from torch.utils.data import Dataset, DistributedSampler
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper


class AutoEpochDistributedSampler(DistributedSampler):
    def __iter__(self) -> Iterator[int]:
        self.epoch += 1
        return super().__iter__()


def DistributedDataset(
    sequence: Dataset[_T], shuffle: bool, seed=0, drop_last=True,
    num_replicas: int = None, rank: int = None
) -> IterDataPipe[_T]:
    return IterableWrapper(AutoEpochDistributedSampler(
        sequence, shuffle=shuffle, seed=seed, drop_last=drop_last,
        num_replicas=num_replicas, rank=rank
    ), deepcopy=False).map(lambda i: sequence[i])  # https://github.com/python/cpython/issues/117735
