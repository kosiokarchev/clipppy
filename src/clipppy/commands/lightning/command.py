from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Generic, get_type_hints, Iterable, Literal, Type, TYPE_CHECKING, TypeVar, Union

import attr
from frozendict import frozendict
from torch.optim import Adam

from phytorchx.attrs import AttrsModule
from .config import BaseSchedulerConfig, Config, DataLoaderConfig, DatasetConfig, OptimizerConfig, SchedulerConfig
from .hyper import nested_iterables, Hyperparams
from .loss import BaseSBILoss
from .patches import LightningModule
from .. import Command
from ... import clipppy
from ...sbi._typing import DEFAULT_LOSS_NAME, DEFAULT_VAL_NAME, SBIBatch, _KT
from ...sbi.data import SBIDataset, AbstractSBIDataset
from ...sbi.data.clipppy_data import ClipppyDataset
from ...sbi.nn import _HeadOoutT, _TailOutT, BaseSBIHead, BaseSBITail
from ...utils import Sentinel

_AbstractLossT = TypeVar('_AbstractLossT')
_LossT = TypeVar('_LossT', bound=BaseSBILoss)


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class AbstractLightningSBICommand(AttrsModule, LightningModule, Command, Generic[_AbstractLossT], ABC):
    commander: clipppy.Clipppy = None

    @classmethod
    def get_type_hints(cls):
        return dict(super().get_type_hints().items() - get_type_hints(LightningModule).items())

    obs_names: Iterable[str] = ()

    # OPTIMIZER
    # ---------

    """Learning rate (passed to the optimizer)."""
    lr: Union[float, Literal[Sentinel.skip]] = 1e-3

    optimizer_config: OptimizerConfig = attr.ib(factory=lambda: OptimizerConfig(Adam, kwargs=dict(fused=True)))

    @property
    def optimizer(self):
        return self.optimizer_config(self.parameters(), lr=self.lr)

    # SCHEDULER
    # ---------

    scheduler_config: BaseSchedulerConfig = attr.ib(factory=SchedulerConfig)

    @property
    def scheduler(self):
        return self.scheduler_config(self.optimizer)

    def configure_optimizers(self):
        return ([sched['scheduler'].optimizer], [sched]) if (sched := self.scheduler) else self.optimizer


    def set_training(self, hp: Hyperparams, max_batch: int = float('inf')):
        # Learning rate
        self.lr = hp.training.lr

        # Optimizer
        if hp.training.optimizer:
            self.optimizer_config = hp.training.optimizer.make()

        # Scheduler
        if hp.training.scheduler:
            self.scheduler_config = hp.training.scheduler.make()

        # Batch size
        assert (hp.training.batch_size < max_batch
                or not hp.training.batch_size % max_batch)
        memory_batch_size = min(hp.training.batch_size, max_batch)
        accumulate_grad_batches = hp.training.batch_size // memory_batch_size

        return memory_batch_size, accumulate_grad_batches

    # LOSS
    # ----

    loss_config: Config[_AbstractLossT]

    @property
    def lossfunc(self) -> _AbstractLossT:
        return self.loss_config()


    _loss_name = DEFAULT_LOSS_NAME
    _val_name = DEFAULT_VAL_NAME

    def log_loss(self, loss, tree=None, loss_name=None):
        losses = {(loss_name := loss_name or self._loss_name): loss}
        if tree is not None:
            losses.update({
                '/'.join(map(str, key)): val.mean()
                for key, val in nested_iterables(tree, keys=(loss_name,))
            })
        self.log_dict(losses, prog_bar=True, logger=True, sync_dist=True)


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class LightningSBICommand(AbstractLightningSBICommand[_LossT], Generic[_LossT, _TailOutT, _HeadOoutT, _KT]):
    head: BaseSBIHead[_HeadOoutT, _KT] = None
    tail: BaseSBITail[_HeadOoutT, _TailOutT, _KT] = None

    def forward(self, batch: SBIBatch, *, head_kwargs=frozendict(), tail_kwargs=frozendict()) -> _TailOutT:
        return self.tail(*self.head(batch.params, batch.obs, **head_kwargs), **tail_kwargs)

    if TYPE_CHECKING:
        __call__ = forward

    # DATASET
    # -------

    param_names: Iterable[str] = ()

    dataset_cls: Type[AbstractSBIDataset] = SBIDataset
    dataset_config: DatasetConfig = attr.ib(factory=lambda: DatasetConfig(ClipppyDataset))

    @cached_property
    def raw_dataset(self):
        return self.dataset_config(config=self.commander)

    def _dataset(self, raw_dataset):
        return self.dataset_cls(raw_dataset, param_names=self.param_names, obs_names=self.obs_names)

    @cached_property
    def dataset(self):
        return self._dataset(self.raw_dataset)

    # LOADER
    # ------

    loader_config: DataLoaderConfig = attr.ib(factory=lambda: DataLoaderConfig(kwargs=dict(batch_size=None)))

    @property
    def loader(self):
        return self.loader_config(self.dataset)

    def _training_loader(self, dataset):
        return self.loader_config(dataset)

    @property
    def training_loader(self):
        return self._training_loader(self.dataset)

    # OPTIMIZER
    # ---------

    @property
    def optimizer(self):
        return self.optimizer_config([
            {'params': self.head.parameters()},
            {'params': self.tail.parameters()},
        ], lr=self.lr)


    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        # checkpoint['clipppy_state'] = self.__getstate__()
        checkpoint['clipppy_nets'] = (self.head, self.tail)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        # self.__setstate__(checkpoint['clipppy_state'])
        self.head, self.tail = checkpoint['clipppy_nets']

    def set_training(self, hp: Hyperparams, max_batch: int = float('inf')):
        memory_batch_size, accumulate_grad_batches = super().set_training(hp, max_batch)
        self.dataset_config.kwargs['batch_size'] = memory_batch_size
        return memory_batch_size, accumulate_grad_batches
