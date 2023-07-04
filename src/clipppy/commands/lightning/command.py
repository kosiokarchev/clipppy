from __future__ import annotations

from typing import Any, Generic, get_type_hints, Iterable, Literal, Type, TYPE_CHECKING, TypeVar, Union

from frozendict import frozendict
from torch.optim import Adam

from .config import BaseSchedulerConfig, Config, DataLoaderConfig, DatasetConfig, OptimizerConfig, SchedulerConfig
from .hyper import nested_iterables, Hyperparams
from .loss import BaseSBILoss
from .patches import LightningModule
from .. import Command
from ... import clipppy
from ...sbi._typing import DEFAULT_LOSS_NAME, DEFAULT_VAL_NAME, SBIBatch, _KT
from ...sbi.data import ClipppyDataset, SBIDataset
from ...sbi.nn import _HeadOoutT, _HeadPoutT, _TailOutT, BaseSBIHead, BaseSBITail
from ...utils import Sentinel


_LossT = TypeVar('_LossT', bound=BaseSBILoss)


class LightningSBICommand(Command, LightningModule, Generic[_TailOutT, _LossT]):
    @classmethod
    def get_type_hints(cls):
        return dict(super().get_type_hints().items() - get_type_hints(LightningModule).items())

    commander: clipppy.Clipppy

    head: BaseSBIHead[_HeadPoutT, _HeadOoutT, _KT]
    tail: BaseSBITail[_HeadPoutT, _HeadOoutT, _TailOutT]

    def forward(self, batch: SBIBatch, *, head_kwargs=frozendict(), tail_kwargs=frozendict()) -> _TailOutT:
        return self.tail(*self.head(batch.params, batch.obs, **head_kwargs), **tail_kwargs)

    if TYPE_CHECKING:
        __call__ = forward

    """Learning rate (passed to the optimizer)."""
    lr: Union[float, Literal[Sentinel.skip]] = 1e-3

    # DATASET
    # -------

    param_names: Iterable[str] = ()
    obs_names: Iterable[str] = ()

    dataset_cls: Type[SBIDataset] = SBIDataset
    dataset_config: DatasetConfig = DatasetConfig(ClipppyDataset)

    @property
    def raw_dataset(self):
        return self.dataset_config(config=self.commander)

    def _dataset(self, raw_dataset):
        return self.dataset_cls(raw_dataset, param_names=self.param_names, obs_names=self.obs_names)

    @property
    def dataset(self):
        return self._dataset(self.raw_dataset)

    # LOADER
    # ------

    loader_config: DataLoaderConfig = DataLoaderConfig(kwargs=dict(batch_size=None))

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

    optimizer_config: OptimizerConfig = OptimizerConfig(Adam)

    @property
    def optimizer(self):
        return self.optimizer_config([
            {'params': self.head.parameters()},
            {'params': self.tail.parameters()},
        ], lr=self.lr)

    # SCHEDULER
    # ---------

    scheduler_config: BaseSchedulerConfig = SchedulerConfig()

    @property
    def scheduler(self):
        return self.scheduler_config(self.optimizer)

    def configure_optimizers(self):
        return ([sched['scheduler'].optimizer], [sched]) if (sched := self.scheduler) else self.optimizer

    # LOSS
    # ----

    loss_config: Config[_LossT]

    @property
    def lossfunc(self) -> _LossT:
        return self.loss_config()

    _loss_name = DEFAULT_LOSS_NAME
    _val_name = DEFAULT_VAL_NAME
    _log_loss_kwargs = dict(prog_bar=False, logger=True)

    def log_loss(self, loss, tree=None, loss_name=None):
        self.log((loss_name := loss_name or self._loss_name), loss, **self._log_loss_kwargs)
        if tree is not None:
            self.log_dict({
                '/'.join(map(str, key)): val.mean()
                for key, val in nested_iterables(tree, keys=(loss_name,))
            }, **self._log_loss_kwargs)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        checkpoint['clipppy_nets'] = (self.head, self.tail)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        self.head, self.tail = checkpoint['clipppy_nets']

    def set_training(self, hp: Hyperparams, max_batch: int):
        # Learning rate
        self.lr = hp.training.lr

        # Scheduler
        if hp.training.scheduler:
            self.scheduler_config = hp.training.scheduler.make()

        # Batch size
        assert (hp.training.batch_size < max_batch
                or not hp.training.batch_size % max_batch)
        memory_batch_size = min(hp.training.batch_size, max_batch)
        accumulate_grad_batches = hp.training.batch_size // memory_batch_size
        self.dataset_config.kwargs['batch_size'] = memory_batch_size

        return memory_batch_size, accumulate_grad_batches
