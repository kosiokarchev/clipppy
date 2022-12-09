from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from more_itertools import always_iterable
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ..nre.validate import MultiNREValidator
from ...sbi._typing import MultiSBIProtocol
from ...utils.plotting.nre import multi_posterior, MultiNREPlotter


@dataclass
class EveryNStepsCallback(Callback):
    every_n_steps: int

    def should_run(self, pl_module: LightningModule):
        return not pl_module.global_step % self.every_n_steps


class TensorBoardCallback(Callback):
    @staticmethod
    def get_summary_writer(pl_module: LightningModule):
        assert isinstance(pl_module.logger, TensorBoardLogger)
        return pl_module.logger.experiment


@dataclass
class DiagnosticfigureCallback(EveryNStepsCallback, TensorBoardCallback, ABC):
    _steps_done: set[int] = field(default_factory=set, init=False)

    @abstractmethod
    def _run(self, *, pl_module: LightningModule, tb: SummaryWriter, global_step: int, **kwargs): ...

    def run(self, trainer: Trainer, pl_module: LightningModule):
        global_step = pl_module.global_step
        if global_step not in self._steps_done:
            self._steps_done.add(global_step)
            return self._run(
                trainer=trainer, pl_module=pl_module,
                tb=self.get_summary_writer(pl_module),
                global_step=global_step
            )

    on_train_end = run

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, *args, **kwargs):
        if self.should_run(pl_module):
            self.run(trainer, pl_module)


@dataclass
class MultiValidationCallback(DiagnosticfigureCallback):
    nre: MultiSBIProtocol
    validator: MultiNREValidator

    validate_name: str = 'validate'
    qq_name: str = 'qq'
    norm_like_name: str = 'norm_like'
    norm_post_name: str = 'norm_post'


    def _run(self, *, tb: SummaryWriter, global_step: int, **kwargs):
        qqfig, likefig, postfig = self.validator(self.nre.head, self.nre.tail)
        tb.add_figure(f'{self.validate_name}/{self.qq_name}', qqfig, global_step)
        tb.add_figure(f'{self.validate_name}/{self.norm_like_name}', likefig, global_step)
        tb.add_figure(f'{self.validate_name}/{self.norm_post_name}', postfig, global_step)


@dataclass
class MultiPosteriorCallback(DiagnosticfigureCallback):
    nre: MultiSBIProtocol
    nrep: MultiNREPlotter
    trace: Mapping[str, Tensor]

    posterior_name: str = 'posterior'

    def _run(self, *, tb: SummaryWriter, global_step: int, **kwargs):
        for key, fig in multi_posterior(self.nre, self.nrep, self.trace).items():
            tb.add_figure(f'{self.posterior_name}/' + '_&_'.join(always_iterable(key)), fig, global_step)
