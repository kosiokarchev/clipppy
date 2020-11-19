import os
import typing as tp

from .yaml import MyYAML

__all__ = ['load_config']


def load_config(path_or_stream: tp.Union[os.PathLike, str, tp.TextIO],
                base_dir=None, force_templating=True, **kwargs):
    return MyYAML(base_dir=base_dir).load(path_or_stream, force_templating=force_templating, **kwargs)
