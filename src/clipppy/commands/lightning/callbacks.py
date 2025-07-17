from __future__ import annotations

from abc import ABC, abstractmethod
from functools import singledispatchmethod
from itertools import chain
from typing import Any, Mapping, Iterable, Union, cast

import attr
import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from matplotlib import pyplot as plt
from more_itertools import always_iterable
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, Logger, WandbLogger
from torch import Size

from .utils import if_not_sanity_checking
from ...sbi._typing import MultiSBIProtocol, _MultiKT, SBIBatch, MultiNREProtocol, MultiNPEProtocol, _SBIObsT
from ...sbi.validate import MultiSBIValidator
from ...utils.plotting.sbi import MultiSBIPosteriorPlotter, MultiSBIValidationPlotter


@attr.define(kw_only=True, slots=False)
class PeriodicCallback(Callback, ABC):
    """Simple checkpoint which happens at reasonable intervals. Modelled after
    `~pytorch_lightning.ModelCheckpoint`."""

    _every_n_train_steps: int = False
    _every_n_epochs: int = 1
    _on_train_epoch: bool = False
    _on_validation: bool = False

    @abstractmethod
    def __call__(self, *, global_step: int, **kwargs): ...

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int):
        if self._every_n_train_steps and trainer.global_step % self._every_n_train_steps == 0:
            self.__call__(trainer=trainer, pl_module=pl_module, global_step=trainer.global_step)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self._on_train_epoch and self._every_n_epochs and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
            self.__call__(trainer=trainer, pl_module=pl_module, global_step=trainer.global_step)

    @if_not_sanity_checking
    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        if self._on_validation:
            self.__call__(trainer=trainer, pl_module=pl_module, global_step=trainer.global_step)


@attr.define(kw_only=True, slots=False)
class DiagnosticFigureMixin:
    logger: Union[Logger, Iterable[Logger]] = None

    @singledispatchmethod
    def _log_figure(self, logger: Logger, name: str, fig, global_step: int):
        raise NotImplementedError

    @_log_figure.register
    def _(self, logger: TensorBoardLogger, name: str, fig, global_step: int):
        logger.experiment.add_figure(name, fig, global_step)

    @_log_figure.register
    def _(self, logger: WandbLogger, name: str, fig, global_step: int):
        # import wandb
        # logger.experiment.log({name: wandb.Image(fig), 'trainer/global_step': global_step}, step=global_step)
        logger.log_image(name, [fig], step=global_step)

    @rank_zero_only
    def log_figure(self, name: str, fig: plt.Figure, global_step: int):
        for logger in always_iterable(self.logger):
            self._log_figure(logger, name, fig, global_step)


@attr.define(kw_only=True, slots=False)
class MultiSBIDiagnosticFigureCallback(DiagnosticFigureMixin, Callback):
    net: MultiSBIProtocol = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.logger = trainer.loggers
        self.net = cast(MultiSBIProtocol, pl_module)


@attr.define(slots=False, kw_only=True)
class MultiSBIPosteriorCallback(MultiSBIDiagnosticFigureCallback, PeriodicCallback):
    data: _SBIObsT
    groups_global: Iterable[_MultiKT] = ()
    groups_local: Iterable[_MultiKT] = ()

    device: Union[str, torch.device] = None

    ref_plotters: Iterable[MultiSBIPosteriorPlotter] = ()

    posterior_name: str = 'posterior'
    corner_kwargs: Mapping[str, Any] = {}
    local_v_truth_kwargs: Mapping[str, Any] = attr.field(default={}, converter=dict(cred=0.68).__or__)

    def _get_name(self, group: _MultiKT):
        return f'{self.posterior_name}/' + '_&_'.join(always_iterable(group))

    @abstractmethod
    def get_wplotter(self, *args, **kwargs): ...

    def forward(self) -> tuple[Mapping[_MultiKT, plt.Figure], Mapping[_MultiKT, plt.Figure]]:
        self.net.head.eval().to(device=self.device), self.net.tail.eval().to(device=self.device)

        wplotter = self.get_wplotter()

        global_figs, local_figs = {}, {}
        for group in self.groups_global:
            global_figs[group], axs = wplotter.corner(group, **self.corner_kwargs)

            for ref_plotter in self.ref_plotters:
                ref_plotter.corner(
                    group, axs=axs,
                    levels=(0.39346934, 0.86466472),
                    plot_prior=False, plot_hist1d=False, plot_hist2d=False, plot_truth=False, plot_bounds=False,
                    # post_kwargs=dict(label='ref', color='k'),
                    post1d_kwargs=dict(linestyle='--'),
                    post2d_kwargs=dict(linestyles='--')
                )

        for group in self.groups_local:
            local_figs[group] = plt.figure()
            ax = wplotter.local_v_truth(group, **self.local_v_truth_kwargs)

            for ref_plotter in self.ref_plotters:
                ref_plotter.local_v_truth(group, cred=self.local_v_truth_kwargs['cred'], ax=ax)

        return global_figs, local_figs

    def __call__(self, *, global_step: int, **kwargs):
        global_figs, local_figs = self.forward()
        for group, fig in chain(global_figs.items(), local_figs.items()):
            self.log_figure(self._get_name(group), fig, global_step)
            plt.close(fig)

    @staticmethod
    def subtype(protocol_type: type[MultiSBIProtocol]) -> type[MultiSBIPosteriorCallback]:
        return {
            MultiNREProtocol: MultiNREPosteriorCallback,
            MultiNPEProtocol: MultiNPEPosteriorCallback,
        }[protocol_type]


@attr.define(slots=False, kw_only=True)
class MultiNREPosteriorCallback(MultiSBIPosteriorCallback):
    plotter: MultiSBIPosteriorPlotter
    net: MultiNREProtocol = None

    def get_wplotter(self, *args, **kwargs):
        return self.plotter.eval_nre((*self.groups_global, *self.groups_local), self.net, self.data)


@attr.define(slots=False, kw_only=True)
class MultiNPEPosteriorCallback(MultiSBIPosteriorCallback):
    nsamples: int = 1000
    plotter_kwargs: Mapping[str, Any] = {}
    net: MultiNPEProtocol = None

    def get_wplotter(self, *args, **kwargs):
        res = self.net.posterior(self.data)
        return MultiSBIPosteriorPlotter(samples=dict(
            (key, val.squeeze(-1)) for keys, dist in res.items()
            for vals in [dist.sample(Size((self.nsamples,)))]
            for key, val in zip(always_iterable(keys), (vals.unsqueeze(-1) if not dist.event_shape else vals).unbind(-1))
        ), **self.plotter_kwargs).with_ratios({
            key: torch.zeros(self.nsamples) for key in res.keys()
        })


@attr.define(slots=False)
class MultiSBIValidationCallback(MultiSBIDiagnosticFigureCallback, PeriodicCallback):
    validator: MultiSBIValidator
    plotter: MultiSBIValidationPlotter
    dataset: Iterable[SBIBatch]
    groups: Iterable[_MultiKT]

    validate_name: str = 'validate'

    pp_name: str = 'pp'
    pp_fig_kwargs: Mapping[str, Any] = attr.field(default={}, converter=dict(figsize=(4, 4)).__or__)
    pp_kwargs: Mapping[str, Any] = {}

    norm_like_name: str = 'norm_like'
    norm_like_fig_kwargs: Mapping[str, Any] = {}
    norm_like_kwargs: Mapping[str, Any] = {}

    norm_post_name: str = 'norm_post'
    norm_post_fig_kwargs: Mapping[str, Any] = {}
    norm_post_kwargs: Mapping[str, Any] = {}

    def forward(self):
        self.net.head.eval(), self.net.tail.eval()
        norm_post, norm_like, creds = self.validator.validate(self.groups, self.net, self.dataset)

        plt.figure(**self.pp_fig_kwargs)
        return self.plotter.pp(creds, **self.pp_kwargs).figure

    def __call__(self, *, global_step: int, **kwargs):
        pp_fig = self.forward()
        self.log_figure(f'{self.validate_name}/{self.pp_name}', pp_fig, global_step)
