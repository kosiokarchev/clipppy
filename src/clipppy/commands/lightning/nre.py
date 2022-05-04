from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, Union

from torch.optim import Adam
from torch.utils._pytree import tree_unflatten

from .config import BaseSchedulerConfig, Config, DataLoaderConfig, DatasetConfig, OptimizerConfig, SchedulerConfig
from .loss import NRELoss
from .patches import LightningModule
from .. import Command
from ..nre import ClipppyDataset
from ..nre._typing import _BatchT, DEFAULT_LOSS_NAME
from ..nre.data import DoublePipe, NREDataset
from ..nre.nn import BaseNREHead, BaseNRETail
from ... import clipppy
from ...utils import Sentinel


class NRE(Command, LightningModule):
    commander: clipppy.Clipppy

    head: BaseNREHead
    tail: BaseNRETail

    def forward(self, batch: _BatchT):
        return self.tail(*self.head(*batch))

    """Learning rate (passed to the optimizer)."""
    lr: Union[float, Literal[Sentinel.skip]] = 1e-3

    param_names: Iterable[str] = ()
    obs_names: Iterable[str] = ()

    dataset_config: DatasetConfig = DatasetConfig(ClipppyDataset)

    @property
    def raw_dataset(self):
        return self.dataset_config(config=self.commander)

    @property
    def dataset(self):
        return NREDataset(self.raw_dataset, param_names=self.param_names, obs_names=self.obs_names)

    loader_config: DataLoaderConfig = DataLoaderConfig(kwargs=dict(batch_size=None))

    @property
    def loader(self):
        return self.loader_config(self.dataset)

    @property
    def training_loader(self):
        return self.loader_config(DoublePipe(self.dataset))

    optimizer_config: OptimizerConfig = OptimizerConfig(Adam)

    @property
    def optimizer(self):
        return self.optimizer_config([
            {'params': self.head.parameters()},
            {'params': self.tail.parameters()},
        ], lr=self.lr)

    scheduler_config: BaseSchedulerConfig = SchedulerConfig()

    @property
    def scheduler(self):
        return self.scheduler_config(self.optimizer)

    loss_config: Config[NRELoss] = Config(NRELoss(), Sentinel.no_call)

    @property
    def lossfunc(self) -> NRELoss:
        return self.loss_config()

    _loss_name = DEFAULT_LOSS_NAME

    def configure_optimizers(self):
        return ([sched['scheduler'].optimizer], [sched]) if (sched := self.scheduler) else self.optimizer

    def training_step(self, batches: tuple[_BatchT, _BatchT], *args, **kwargs):
        theta_1, x_1 = self.head(*batches[0])
        theta_2, x_2 = self.head(*batches[1])

        ret_1 = self.lossfunc(self.tail(theta_1, x_1), self.tail(theta_2, x_1))
        ret_2 = self.lossfunc(self.tail(theta_2, x_2), self.tail(theta_1, x_2))

        spec = ret_1.spec
        assert spec == ret_2.spec

        if spec is not None and isinstance(tree := tree_unflatten([
            (a + b) / 2 for a, b in zip(ret_1.flat, ret_2.flat)
        ], ret_1.spec), Mapping):
            self.log_dict({
                f'{self._loss_name}/{key}': val.mean() for key, val in tree.items()
            }, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        loss = (ret_1.loss + ret_2.loss) / 2
        self.log(self._loss_name, loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        return loss


    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        checkpoint['clipppy_nets'] = (self.head, self.tail)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        self.head, self.tail = checkpoint['clipppy_nets']
