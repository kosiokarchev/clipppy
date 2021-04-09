import os
from types import FrameType
from typing import Any, AnyStr, Mapping, TextIO, Union

from .clipppy import Clipppy
from .yaml import ClipppyYAML


__all__ = 'load_config', 'Clipppy', 'ClipppyYAML'


def load_config(path_or_stream: Union[os.PathLike, str, TextIO],
                base_dir: Union[os.PathLike, AnyStr] = None,
                interpret_as_Clipppy=True, force_templating=True,
                scope: Union[Mapping[str, Any], FrameType, int] = 0,
                **kwargs) -> Union[Clipppy, Any]:
    return (ClipppyYAML(base_dir=base_dir, interpret_as_Clipppy=interpret_as_Clipppy)
            .load(path_or_stream, force_templating=force_templating,
                  scope=scope+1 if isinstance(scope, int) else scope, **kwargs))
