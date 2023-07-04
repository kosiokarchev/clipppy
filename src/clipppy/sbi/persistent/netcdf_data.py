from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, InitVar, field
from functools import partial
from typing import Collection, Union, Mapping, Iterator, Sequence, Optional, Iterable

import netCDF4 as nc
import numpy as np
import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate_fn_map

from . import PersistentDataset
from ..data import _ValuesT


class VLTensor(Tensor):
    pass


# noinspection PyUnusedLocal
@partial(default_collate_fn_map.__setitem__, VLTensor)
def collate_vltensor(batch: Iterable[VLTensor], *args, **kwargs):
    return list(el.as_subclass(Tensor) for el in batch)
    # return torch.nested.as_nested_tensor(list(el.as_subclass(Tensor) for el in batch))


@dataclass
class NetCDFDataset(PersistentDataset):
    store: InitVar[Union[str, nc.Dataset]]
    mode: InitVar[str] = 'r'
    keys: Optional[Collection[str]] = None

    index_name: str = 'index'
    var_dimensions: Mapping[str, tuple[str]] = field(default_factory=lambda: defaultdict(tuple))

    def __post_init__(self, store: Union[str, nc.Dataset], mode: str):
        self.group = store if isinstance(store, nc.Dataset) else nc.Dataset(store, mode)
        self.group.set_auto_mask(False)

        if self.index_name not in self.group.dimensions:
            self.group.createDimension(self.index_name)
        self.index: nc.Dimension = self.group.dimensions[self.index_name]

    def _get_vltype(self, dtype: np.dtype):
        for vlt in self.group.vltypes.values():
            if vlt.dtype is dtype:
                return vlt
        return self.group.createVLType(dtype, 'vl_'+dtype.name)

    def _extend_batch(self, values: Mapping[str, Sequence[Tensor]]):
        i0 = len(self)
        for key, val in values.items():
            if isinstance(val, Tensor) and val.is_nested:
                val = val.unbind()

            if isinstance(val, Tensor):
                val = val.numpy(force=True)
                dtype = val.dtype
            else:
                # val = np.array([v.numpy(force=True) for v in val], dtype=object)
                val = np.array([
                    _.numpy(force=True)
                    for v in val
                    for _ in (v.unsqueeze(-1) if v.ndim == 1 else v).flatten(1).movedim(0, -1)
                ], dtype=object).reshape(len(val), *val[0].shape[1:])
                dtype = self._get_vltype(val.flat[0].dtype)

            if key not in self.group.variables:
                v = self.group.createVariable(key, dtype, (self.index_name, *self.var_dimensions[key]), fill_value=False)
                v.set_auto_mask(False)

            self.group.variables[key][i0:] = val

    @property
    def variables(self) -> Iterator[tuple[str, nc.Variable]]:
        return (self.group.variables.items() if self.keys is None else
                ((key, self.group[key]) for key in self.keys))

    def __getitem__(self, item) -> _ValuesT:
        return {key: torch.tensor(val[item], device='cpu').as_subclass(
            VLTensor if isinstance(val.datatype, nc.VLType) else Tensor
        ) for key, val in self.variables}

    def __len__(self):
        return len(self.index)
