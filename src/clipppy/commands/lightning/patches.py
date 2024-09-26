from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, InitVar
from functools import partial
from pathlib import Path
from typing import MutableMapping, Mapping, Union, Type, Any

import pytorch_lightning as pl
import pytorch_lightning.loggers
from frozendict import frozendict
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

for _mname in 'pytorch_lightning.utilities.logger', 'lightning_fabric.utilities.logger':
    try:
        _add_prefix = __import__(_mname, globals(), locals(), ['_add_prefix'], 0)._add_prefix
        break
    except (ImportError, AttributeError):
        pass


class LightningModule(pl.LightningModule):
    # noinspection PyAttributeOutsideInit
    def just_save_hyperparameters(self, hparams: MutableMapping, logger: bool = True):
        self._log_hyperparams = logger
        self._set_hparams(hparams)
        self._hparams_name = 'kwargs'
        self._hparams_initial = deepcopy(self._hparams)


class WandbLogger(pl.loggers.WandbLogger):
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int = None, **kwargs) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = dict(_add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR))
        if step is not None:
            metrics['trainer/global_step'] = step
        self.experiment.log(metrics, step=step, **kwargs)


@dataclass
class WandbHooker(Callback):
    project: str
    name: str
    resume_last: bool = False
    username: str = None

    running: bool = field(init=False, default=False)
    kwargs: InitVar[Mapping[str, Any]] = field(default=frozendict())

    @rank_zero_only
    def __post_init__(self, kwargs):
        import wandb
        self.wandb = wandb
        self.run = self.wandb.init(
            project=self.project, name=self.name, **{**dict(
                resume='allow', sync_tensorboard=True,
                id=self.wandb.Api().runs(path=f'{self.username}/{self.project}')[0].id if self.resume_last else None
            ), **kwargs}
        )
        self.running = True

    @rank_zero_only
    def _finish(self):
        if self.running:
            self.wandb.finish()
            self.running = False

    def on_exception(self, trainer, pl_module, exception: BaseException):
        if isinstance(exception, KeyboardInterrupt):
            self._finish()

    def on_fit_end(self, *args, **kwargs):
        self._finish()



Trainer: Type[pl.Trainer] = partial(pl.Trainer, log_every_n_steps=1, enable_model_summary=False, max_epochs=-1)
ModelCheckpoint: Type[pl.callbacks.ModelCheckpoint] = partial(
    pl.callbacks.ModelCheckpoint,
    every_n_epochs=1, save_on_train_epoch_end=False,  # save on validation
    save_top_k=-1, save_last=True, filename='{step}',
    monitor='val'
)


def get_best_model_path(logdir: Union[str, Path], normalize=True):
    from ruamel import yaml

    logdir = Path(logdir)
    ckpt_path = Path(min(
        yaml.safe_load((logdir / 'ckpt_vals.yaml').open()).items(),
        key=lambda keyval: keyval[1]
    )[0])
    return logdir.joinpath(ckpt_path.relative_to(ckpt_path.parents[1])) if normalize else ckpt_path
