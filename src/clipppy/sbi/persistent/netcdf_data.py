from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, InitVar, field
from pathlib import Path
from typing import Collection, Union, Mapping, Iterator, Sequence, Optional, Container

import netCDF4 as nc
import numpy as np
import torch
import xarray as xa
from torch import Tensor

from . import PersistentDataset
from ...utils.dataframe import _KT, AbstractTensorDataFrame
from ...utils.dataframe.vltensor import VLTensor
from ...utils.typing import _Tensor_like


@dataclass
class NetCDFDataset(PersistentDataset):
    store: InitVar[Union[str, Path, nc.Dataset]]
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
            if isinstance(val, VLTensor):
                val = [val.as_subclass(Tensor)]

            if isinstance(val, Tensor):
                val = val.numpy(force=True)
                dtype = val.dtype
            else:
                out = np.empty((len(val),), dtype=object)
                out[:] =[
                    _.numpy(force=True)
                    for v in val
                    for _ in (v.unsqueeze(-1) if v.ndim == 1 else v).flatten(1).movedim(0, -1)
                ]
                val = out.reshape(len(val), *val[0].shape[1:])
                dtype = self._get_vltype(val.flat[0].dtype)

            if key not in self.group.variables:
                v = self.group.createVariable(key, dtype, (self.index_name, *self.var_dimensions[key]), fill_value=False)
                v.set_auto_mask(False)

            self.group.variables[key][i0:] = val

    @property
    def variables(self) -> Iterator[tuple[str, nc.Variable]]:
        return (self.group.variables.items() if self.keys is None else
                ((key, self.group[key]) for key in self.keys))

    def __len__(self):
        return len(self.index)


@dataclass
class NetCDFDataFrame(AbstractTensorDataFrame, NetCDFDataset):
    device: torch.device = field(default=None, kw_only=True)

    def _to_tensor_like(self, val, vltype=False):
        return (
            list(torch.tensor(v.item() if v.dtype.kind == 'O' else v, device=self.device) for v in res)
            if (res := np.array(val)).dtype.kind == 'O' else
            torch.tensor(res, device=self.device).as_subclass(
                VLTensor if vltype else Tensor
            )
        )

    def _getitem(self, item) -> Mapping[_KT, Tensor]:
        return {key: self._to_tensor_like(val[item], isinstance(val.datatype, nc.VLType)) for key, val in self.variables}

    def _getitem_column(self, item: str) -> _Tensor_like:
        return self._to_tensor_like(self.group[item])

    __len__ = NetCDFDataset.__len__


@dataclass
class XDataFrame(AbstractTensorDataFrame):
    xd: xa.Dataset

    index_name: str = 'index'
    vlnames: Container[str] = ()

    device: torch.device = field(default=None, kw_only=True)

    def __len__(self):
        return len(self.xd[self.index_name])

    def _to_tensor_like(self, val, vl=False):
        return (
            list(torch.tensor(v.item() if v.dtype.kind == 'O' else v, device=self.device) for v in res)
            if (res := np.array(val)).dtype.kind == 'O' else
            torch.tensor(res, device=self.device).as_subclass(
                VLTensor if vl else Tensor
            )
        )

    def _getitem(self, item) -> Mapping[_KT, Tensor]:
        return {key: self._to_tensor_like(val[item], key in self.vlnames) for key, val in self.xd.data_vars}

    def _getitem_column(self, item: str) -> _Tensor_like:
        return self._to_tensor_like(self.xd[item])


# ds = xa.open_mfdataset(
#     'train/resset2-flat-2000/resset2-flat-2000-1_*.nc',
#     chunks={}, combine='nested', concat_dim=['index'], parallel=True)
