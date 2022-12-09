from __future__ import annotations

from functools import partial
from operator import itemgetter
from os import PathLike
from typing import Mapping, Union, Collection, Iterator, Optional, TypeVar
from warnings import warn

import zarr
from more_itertools import all_equal, one, unique_everseen
from torch import Tensor
from torch.utils.data import Dataset

from .data import _OT


_T = TypeVar('_T')


class ZarrDataset(Dataset[_OT]):
    def __init__(self, store: Union[zarr.storage.Store, PathLike, str], keys: Optional[Collection[str]] = None):
        self.group = zarr.hierarchy.open_group(store)
        self.keys = keys

    @staticmethod
    def tensor_to_zarrable(t: Tensor):
        return t.detach().cpu().numpy()

    def get_values_for_keys(self, values: Mapping[str, _T]) -> Mapping[str, _T]:
        return values if self.keys is None else {key: values[key] for key in self.keys}

    def extend_batch(self, values: Mapping[str, Tensor]):
        values = self.get_values_for_keys(values)

        if not all_equal(val.shape[0] for val in values.values()):
            warn(f'Appending unequal-length batches to {type(self).__name__}', RuntimeWarning)

        for key, val in values.items():
            (partial(self.group.__setitem__, key) if key not in self.group
             else self.group[key].append
             )(self.tensor_to_zarrable(val))

    def extend(self, values: Mapping[str, Tensor]):
        self.extend_batch({
            key: val.unsqueeze(0)
            for key, val in self.get_values_for_keys(values).items()})

    @property
    def arrays(self) -> Iterator[tuple[str, zarr.Array]]:
        return (
            self.group.arrays() if self.keys is None else
            ((key, self.group[key]) for key in self.keys)
        )

    def __getitem__(self, item) -> _OT:
        return {key: val[item] for key, val in self.arrays}

    def __len__(self) -> int:
        return one(unique_everseen(map(len, map(itemgetter(1), self.arrays))))
