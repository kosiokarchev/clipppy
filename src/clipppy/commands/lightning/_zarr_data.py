from __future__ import annotations

from functools import partial
from operator import itemgetter
from typing import Mapping
from warnings import warn

import zarr
from more_itertools import all_equal, one, unique_everseen
from torch import Tensor
from torch.utils.data import Dataset

from clipppy.commands.sbi.data import _OT


class ZarrDataset(Dataset[_OT]):
    def __init__(self, store):
        self.group = zarr.hierarchy.open_group(store)

    @staticmethod
    def tensor_to_zarrable(t: Tensor):
        return t.detach().cpu().numpy()

    def append_batch(self, values: Mapping[str, Tensor]):
        if not all_equal(val.shape[0] for val in values.values()):
            warn(f'Appending unequal-length batches to {type(self).__name__}', RuntimeWarning)

        for key, val in values.items():
            (partial(self.group.__setitem__, key) if key not in self.group
             else self.group[key].append
             )(self.tensor_to_zarrable(val))

    def append(self, values: Mapping[str, Tensor]):
        self.append_batch({key: val.unsqueeze(0) for key, val in values.items()})

    def __getitem__(self, item) -> _OT:
        return {key: val[item] for key, val in self.group.arrays()}

    def __len__(self):
        return one(unique_everseen(map(len, map(itemgetter(1), self.group.arrays()))))
