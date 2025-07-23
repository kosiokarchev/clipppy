from functools import wraps
from pathlib import Path
from typing import TypeVar

from pytorch_lightning import Trainer

_T = TypeVar('_T')


def get_best_ckpt(logdir: str | Path, normalize=True):
    from ruamel import yaml

    logdir = Path(logdir)
    ckpt_path = Path(min(
        yaml.YAML(typ='safe', pure=True).load((logdir / 'checkpoints/best_k_models.yaml').open()).items(),
        key=lambda keyval: keyval[1]
    )[0])
    return logdir.joinpath(ckpt_path.relative_to(ckpt_path.parents[1])) if normalize else ckpt_path


def if_not_sanity_checking(f: _T) -> _T:
    @wraps(f)
    def wrap(self, trainer: Trainer, *args, **kwargs):
        if not trainer.sanity_checking:
            return f(self, trainer, *args, **kwargs)
    return wrap
