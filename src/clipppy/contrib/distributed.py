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

        default_device = phytorchx.get_default_device()
        torch.set_default_device('cpu')
        ret = super().__iter__()
        torch.set_default_device(default_device)
        return ret


def DistributedDataset(
    sequence: Dataset[_T], shuffle: bool, batch=None, seed=0, drop_last=True,
    num_replicas: int = None, rank: int = None
) -> IterDataPipe[_T]:
    res = IterableWrapper(AutoEpochDistributedSampler(
        sequence, shuffle=shuffle, seed=seed, drop_last=drop_last,
        num_replicas=num_replicas, rank=rank
    ), deepcopy=False)
    return (res.batch(batch) if batch is not None else res).map(lambda i: sequence[i])
