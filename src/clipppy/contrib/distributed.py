from typing import Iterator

from torch.utils.data import Dataset, DistributedSampler
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from clipppy.utils.typing import _T


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
