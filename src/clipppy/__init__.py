import io
import os
from types import FrameType
from typing import Any, AnyStr, Mapping, TextIO, Union

from .clipppy import Clipppy
from .yaml import ClipppyYAML, determine_scope

__version__ = '0.0.42'
__all__ = 'load_config', 'loads', 'Clipppy', 'ClipppyYAML'


def load_config(path_or_stream: Union[os.PathLike, str, TextIO],
                base_dir: Union[os.PathLike, AnyStr] = None,
                interpret_as_Clipppy=True, force_templating=True,
                scope: Union[Mapping[str, Any], FrameType] = None,
                **kwargs) -> Union[Clipppy, Any]:
    return (ClipppyYAML(base_dir=base_dir, interpret_as_Clipppy=interpret_as_Clipppy)
            .load(path_or_stream, force_templating=force_templating,
                  scope=determine_scope(scope), **kwargs))


def loads(string: str, *, interpret_as_Clipppy=False, **kwargs):
    return load_config(io.StringIO(string),
                       scope=determine_scope(kwargs.pop('scope', None)),
                       interpret_as_Clipppy=interpret_as_Clipppy, **kwargs)
