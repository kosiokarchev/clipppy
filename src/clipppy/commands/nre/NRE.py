from __future__ import annotations

from typing import Callable, Iterable, Literal, Mapping, Type, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from typing_extensions import TypeAlias

from .data import NREDataset, ClipppyDataset
from .loss import NRELoss
from .nn import BaseNREHead, BaseNRETail
from ..optimizing_command import OptimizingCommand
from ... import clipppy
from ...utils import merge_if_not_skip, Sentinel


__all__ = 'NRE',


_OptimizerT: TypeAlias = Optimizer
_LossT: TypeAlias = Callable[[Tensor, Tensor], Tensor]


class NRE(OptimizingCommand[_OptimizerT, _LossT]):
    commander: clipppy.Clipppy

    def _instantiate_optimizer(self, kwargs):
        return self.optimizer_cls([
            {'params': self.head.parameters()},
            {'params': self.tail.parameters()},
        ], **kwargs)

    optimizer_cls = Adam
    loss_cls = NRELoss()
    loss_args = Sentinel.no_call

    dataset_cls: Union[Type[Dataset], Callable[..., Dataset], Dataset] = ClipppyDataset
    dataset_args: Union[Mapping, Literal[Sentinel.no_call]] = {}

    param_names: Iterable[str] = ()
    obs_names: Iterable[str] = ()

    @property
    def dataset(self):
        return NREDataset(self._instantiate(
            self.dataset_cls, self.dataset_args, {'config': self.commander}
        ), param_names=self.param_names, obs_names=self.obs_names)

    dataloader_cls: Union[Type[DataLoader], Callable[..., DataLoader], DataLoader] = DataLoader
    dataloader_args: Union[Mapping, Literal[Sentinel.no_call]] = {}

    batch_size: Union[int, Literal[Sentinel.skip]] = None
    pseudobatch_size: int = 1

    @property
    def loader(self):
        if self.dataloader_args is Sentinel.no_call:
            return self.dataloader_cls
        return self.dataloader_cls(self.dataset, **merge_if_not_skip(self.dataloader_args, {'batch_size': self.batch_size}))

    head: BaseNREHead
    tail: BaseNRETail

    def step(self, loader, optim: _OptimizerT, lossfunc: _LossT, train: Iterable[Module] = None):
        optim.zero_grad(set_to_none=True)
        for mod in (train or (self.head, self.tail)):
            mod.train()

        loss = 0
        for _ in range(self.pseudobatch_size):
            theta_1, x_1 = self.head(*next(loader))
            theta_2, x_2 = self.head(*next(loader))

            loss = loss + (
                lossfunc(self.tail(theta_1, x_1), self.tail(theta_2, x_1))
                + lossfunc(self.tail(theta_2, x_2), self.tail(theta_1, x_2))
            )

        loss = loss / (2 * self.pseudobatch_size)

        if not torch.isnan(loss):
            loss.backward()
            optim.step()

        return loss.item()

    def forward(self, *args, **kwargs):
        return super().forward(loader=iter(self.loader), optim=self.optimizer, lossfunc=self.lossfunc)
