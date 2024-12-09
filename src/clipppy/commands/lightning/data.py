from abc import ABC
from functools import cached_property
from itertools import chain
from typing import Iterable, Mapping, Callable

import attr
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper

from phytorchx.dataframe import AbstractTensorDataFrame, _KT
from .callbacks import MultiSBIValidationCallback
from .command import LightningSBICommand, AbstractLightningSBICommand
from ...sbi._typing import _MultiKT, MultiSBIProtocol
from ...sbi.validate import MultiSBIValidator
from ...utils.plotting.sbi import MultiSBIValidationPlotter, MultiSBIPosteriorPlotter


@attr.s(auto_attribs=True)
class AbstractMultiSBIDataModule(LightningDataModule, ABC):
    def __attrs_pre_init__(self):
        super().__init__()

    sbi: AbstractLightningSBICommand
    batch_size: int

    labels: Mapping = attr.ib(factory=dict, kw_only=True)
    validation_plotter_kwargs: Mapping = attr.ib(factory=dict, kw_only=True)
    posterior_plotter_kwargs: Mapping = attr.ib(factory=dict, kw_only=True)

    @property
    def keys(self):
        return chain(self.sbi.obs_names)

    @cached_property
    def _protocol_type(self):
        return MultiSBIProtocol.resolve(type(self.sbi))

    def __attrs_post_init__(self):
        super().__init__()
        self._plotter_kwargs = dict(labels=self.labels, batch_size=self.batch_size)


@attr.s(auto_attribs=True)
class DataFrameSBIDataModule(AbstractMultiSBIDataModule):
    train_dataset: AbstractTensorDataFrame
    val_dataset: AbstractTensorDataFrame

    train_preprocessor: Callable[[Mapping[_KT, Tensor]], Mapping[_KT, Tensor]] = attr.ib(default=None, kw_only=True)

    def _dataset(self, dataset: AbstractTensorDataFrame, shuffle: bool):
        return dataset.batched(self.batch_size, shuffle=shuffle)

    @staticmethod
    def _ds_to_dl(ds):
        return DataLoader(ds)

    def _dataloader(self, dataset: AbstractTensorDataFrame, shuffle: bool, pre=None):
        ds = self._dataset(dataset, shuffle=shuffle)
        return self._ds_to_dl(ds if pre is None else IterableWrapper(ds).map(pre))


    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True, pre=self.train_preprocessor)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False)


class MultiSBIDataModule(DataFrameSBIDataModule):
    sbi: LightningSBICommand

    def _ds_to_dl(self, ds):
        return self.sbi._training_loader(self.sbi._dataset(ds))

    @property
    def keys(self):
        return chain(self.sbi.param_names, self.sbi.obs_names)

    @cached_property
    def val_params(self):
        return self.val_dataset[list(self.sbi.param_names)]

    @cached_property
    def posterior_plotter(self):
        return MultiSBIPosteriorPlotter(samples=self.val_params, **{**self._plotter_kwargs, **self.posterior_plotter_kwargs})

    @cached_property
    def validation_plotter(self):
        return MultiSBIValidationPlotter(self.val_params, **{**self._plotter_kwargs, **self.validation_plotter_kwargs})

    def validation_callback(self, groups: Iterable[_MultiKT] = None, **kwargs):
        return MultiSBIValidationCallback(
            MultiSBIValidator.subtype(self._protocol_type)(self.val_params, batch_size=self.batch_size),
            self.validation_plotter,
            self.sbi._dataset(self.val_dataset.batched(self.batch_size, shuffle=False)),
            self.sbi.tail.tails.keys() if groups is None else groups,
            **kwargs
        )
