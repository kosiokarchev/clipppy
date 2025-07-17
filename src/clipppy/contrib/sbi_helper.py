import dataclasses
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Generic, Callable, TypeVar, TYPE_CHECKING

import torch
from ruamel import yaml

import phytorchx
from ..commands.lightning.utils import get_best_ckpt

_T = TypeVar('_T')
_O = TypeVar('_O')


@dataclass
class SavedProperty(Generic[_O, _T]):
    pathfunc: Callable[[_O], Path]
    loadfunc: Callable[[Path], _T] = phytorchx.load
    savefunc: Callable[[_T, Path], None] = torch.save

    def __set_name__(self, owner: type[_O], name: str):
        setattr(owner, name+'_name', property(self.pathfunc))

    def __get__(self, instance: _O, owner: type[_O]):
        return self.loadfunc(self.pathfunc(instance))

    def __set__(self, instance: _O, value: _T):
        (path := self.pathfunc(instance)).parent.mkdir(parents=True, exist_ok=True)
        self.savefunc(value, path)

    @classmethod
    def suffixed(cls, base_name: str, suffix: str, ext: str, **kwargs):
        return cls(
            lambda o: getattr(o, base_name).with_stem(getattr(o, base_name).stem + suffix).with_suffix(ext),
            **kwargs)


@dataclass(kw_only=True)
class SBIHelper:
    def clone(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    base_datadir: ClassVar = Path('data')
    base_traindir: ClassVar = Path('train')
    base_logdir: ClassVar = Path('lightning_logs')
    base_resdir: ClassVar = Path('res')

    basename: str = None
    zoom_stage: int = None

    def zoom(self):
        return dataclasses.replace(self, zoom_stage=self.zoom_stage+1)

    def __post_init__(self):
        self.name = f'{self.basename}-{self.zoom_stage}'

        self.traindir = self.base_traindir / self.basename / str(self.zoom_stage)
        self.logdir = self.base_logdir / self.name
        self.resdir = self.base_resdir / self.basename / str(self.zoom_stage)

    if TYPE_CHECKING:
        mock_data_name: ClassVar[Path]
        bestnet_name: ClassVar[Path]
    mock_data = SavedProperty(lambda self: self.base_datadir / f'{self.basename}.pt')
    bestnet = SavedProperty(lambda self: self.resdir / 'bestnet.ckpt')

    @cached_property
    def logger(self):
        from pytorch_lightning.loggers import TensorBoardLogger
        return TensorBoardLogger(self.base_logdir, self.name)

    @cached_property
    def bestnet_version(self):
        return self.logdir / yaml.YAML(typ='safe', pure=True).load((self.base_resdir / 'bestnets.yaml').open())[self.basename][self.zoom_stage]

    @cached_property
    def bestnet_ckpt(self):
        return get_best_ckpt(self.bestnet_version)

    def symlink_best(self):
        self.bestnet_name.parent.mkdir(exist_ok=True, parents=True)
        self.bestnet_name.unlink(missing_ok=True)
        return os.symlink(self.bestnet_ckpt.absolute(), self.bestnet_name)
