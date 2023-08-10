from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Any, Mapping, Iterable, Union

from deprecated import deprecated
from matplotlib import pyplot as plt
from more_itertools import always_iterable
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, Logger, WandbLogger
from torch import Tensor

from ..nre.validate import MultiNREValidator
from ...sbi._typing import MultiSBIProtocol, _MultiKT
from ...utils.plotting.nre.latent import MultiLatentPlotter
from ...utils.plotting.nre.multi import multi_posterior, MultiNREPlotter
from ...utils.plotting.sbi import MultiSBIPlotter


@dataclass
class EveryNStepsCallback(Callback, ABC):
    every_n_steps: int
    _steps_done: set[int] = field(default_factory=set, init=False)

    def should_run(self, pl_module: LightningModule):
        return not (pl_module.global_step+1) % self.every_n_steps

    @abstractmethod
    def _run(self, *, trainer: Trainer, pl_module: LightningModule, global_step: int, **kwargs): ...

    def run(self, trainer: Trainer, pl_module: LightningModule, **kwargs):
        global_step = pl_module.global_step
        if global_step not in self._steps_done:
            self._steps_done.add(global_step)
            return self._run(trainer=trainer, pl_module=pl_module, global_step=global_step, **kwargs)

    # on_train_end = run

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, *args, **kwargs):
        if self.should_run(pl_module):
            self.run(trainer, pl_module, **kwargs)


@dataclass
class DiagnosticFigureMixin:
    logger: Union[Logger, Iterable[Logger]]

    @singledispatchmethod
    def _log_figure(self, logger: Logger, name: str, fig, global_step: int):
        raise NotImplementedError

    @_log_figure.register
    def _(self, logger: TensorBoardLogger, name: str, fig, global_step: int):
        logger.experiment.add_figure(name, fig, global_step)

    @_log_figure.register
    def _(self, logger: WandbLogger, name: str, fig, global_step: int):
        logger.log_image(name, [fig], step=global_step)

    def log_figure(self, name: str, fig, global_step: int):
        for logger in always_iterable(self.logger):
            self._log_figure(logger, name, fig, global_step)


@dataclass
class MultiSBIPosteriorCallback(DiagnosticFigureMixin, EveryNStepsCallback):
    net: MultiSBIProtocol
    plotter: MultiSBIPlotter
    data: Mapping[str, Tensor]
    groups_global: Iterable[_MultiKT] = ()
    groups_latent: Iterable[_MultiKT] = ()

    posterior_name: str = 'posterior'
    corner_kwargs: dict = field(default_factory=dict)
    latent_v_truth_kwargs: dict = field(default_factory=dict)

    def _get_name(self, group: _MultiKT):
        return f'{self.posterior_name}/' + '_&_'.join(always_iterable(group))

    def _run(self, *, global_step: int, **kwargs):
        self.net.head.eval(), self.net.tail.eval()
        wplotter = self.plotter.eval((*self.groups_global, *self.groups_latent), self.net, self.data)

        for group in self.groups_global:
            self.log_figure(self._get_name(group), wplotter.corner(group, **self.corner_kwargs)[0], global_step)

        for group in self.groups_latent:
            fig = plt.figure()
            wplotter.latent_v_truth(group, **{'cred': 0.68, **self.latent_v_truth_kwargs})
            self.log_figure(self._get_name(group), fig, global_step)


@deprecated(f'Use {MultiSBIPosteriorCallback.__name__} instead.')
@dataclass
class MultiPosteriorCallback(DiagnosticFigureMixin, EveryNStepsCallback, ABC):
    nre: MultiSBIProtocol
    nrep: MultiNREPlotter
    trace: Mapping[str, Tensor]

    posterior_name: str = 'posterior'

    def _run(self, *, global_step: int, **kwargs):
        self.nre.head.eval(), self.nre.tail.eval()
        for key, fig in multi_posterior(self.nre, self.nrep, self.trace).items():
            self.log_figure(f'{self.posterior_name}/' + '_&_'.join(always_iterable(key)), fig, global_step)


@deprecated(f'Use {MultiSBIPosteriorCallback.__name__} instead.')
@dataclass
class MultiLatentCallback(DiagnosticFigureMixin, EveryNStepsCallback, ABC):
    nre: MultiSBIProtocol
    plotter: MultiLatentPlotter
    trace: Mapping[str, Tensor]
    groups: Iterable[_MultiKT]

    cred: float = 0.68

    posterior_name: str = 'posterior'

    def _run(self, *, global_step: int, **kwargs):
        self.nre.head.eval(), self.nre.tail.eval()
        wlp = self.plotter.eval(self.groups, self.nre, self.trace)

        for key in self.groups:
            fig = plt.figure()
            wlp.plot_1d(key, self.cred)
            self.log_figure(f'{self.posterior_name}/' + '_&_'.join(always_iterable(key)), fig, global_step)


@dataclass
class MultiValidationCallback(DiagnosticFigureMixin, EveryNStepsCallback, ABC):
    nre: MultiSBIProtocol
    validator: MultiNREValidator

    validate_name: str = 'validate'
    qq_name: str = 'qq'
    norm_like_name: str = 'norm_like'
    norm_post_name: str = 'norm_post'

    def _run(self, *, global_step: int, **kwargs):
        self.nre.head.eval(), self.nre.tail.eval()
        qqfig, likefig, postfig = self.validator(self.nre.head, self.nre.tail)
        self.log_figure(f'{self.validate_name}/{self.qq_name}', qqfig, global_step)
        self.log_figure(f'{self.validate_name}/{self.norm_like_name}', likefig, global_step)
        self.log_figure(f'{self.validate_name}/{self.norm_post_name}', postfig, global_step)
