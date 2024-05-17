from functools import wraps
from operator import itemgetter
from pathlib import Path
from typing import Union, TypeVar

import yaml
from pytorch_lightning import Trainer

_T = TypeVar('_T')


def get_best_ckpt(folder: Union[str, Path]) -> Path:
    return (folder := Path(folder)) / '/'.join(
        Path(min(
            yaml.safe_load((folder / 'ckpt_vals.yaml').open()).items(),
            key=itemgetter(1)
        )[0]).parts[-2:]
    )


def if_not_sanity_checking(f: _T) -> _T:
    @wraps(f)
    def wrap(self, trainer: Trainer, *args, **kwargs):
        if not trainer.sanity_checking:
            return f(self, trainer, *args, **kwargs)
    return wrap
