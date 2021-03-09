import os
import typing as tp

from .Clipppy import Clipppy
from .yaml import ClipppyYAML

__all__ = 'load_config', 'Clipppy', 'ClipppyYAML'


def load_config(path_or_stream: tp.Union[os.PathLike, str, tp.TextIO],
                base_dir: tp.Union[os.PathLike, tp.AnyStr] = None,
                interpret_as_Clipppy=True,
                force_templating=True, **kwargs) -> tp.Union[Clipppy, tp.Any]:
    return (ClipppyYAML(base_dir=base_dir, interpret_as_Clipppy=interpret_as_Clipppy)
            .load(path_or_stream, force_templating=force_templating, **kwargs))
