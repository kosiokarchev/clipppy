from dataclasses import dataclass, field, InitVar
from typing import Mapping, Any

import pytorch_lightning as pl
import pytorch_lightning.loggers
from frozendict import frozendict

LightningModule = pl.LightningModule
ModelCheckpoint = pl.callbacks.ModelCheckpoint
Trainer = pl.Trainer
WandbLogger = pl.loggers.WandbLogger


@dataclass
class WandbHooker(pl.callbacks.Callback):
    project: str
    name: str
    resume_last: bool = False
    username: str = None

    running: bool = field(init=False, default=False)
    kwargs: InitVar[Mapping[str, Any]] = field(default=frozendict())
