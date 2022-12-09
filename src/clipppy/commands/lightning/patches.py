from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import MutableMapping

import pytorch_lightning as pl
import pytorch_lightning.loggers


class LightningModule(pl.LightningModule):
    # noinspection PyAttributeOutsideInit
    def just_save_hyperparameters(self, hparams: MutableMapping, logger: bool = True):
        self._log_hyperparams = logger
        self._set_hparams(hparams)
        self._hparams_name = 'kwargs'
        self._hparams_initial = deepcopy(self._hparams)


class TensorBoardLogger(pl.loggers.TensorBoardLogger):
    @pl.utilities.rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)


Trainer = partial(pl.Trainer, log_every_n_steps=1, enable_model_summary=False)
LearningRateMonitor = partial(
    pl.callbacks.LearningRateMonitor,
    logging_interval='step')
ModelCheckpoint = partial(pl.callbacks.ModelCheckpoint, save_top_k=-1)
