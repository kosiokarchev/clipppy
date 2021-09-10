from os import PathLike
from types import FrameType
from typing import Any, AnyStr, Mapping, TextIO, Union

from .clipppy import Clipppy
from .yaml import ClipppyYAML


__all__ = 'load_config', 'loads', 'Clipppy', 'ClipppyYAML', '__version__'


__version__: str

def load(
    path_or_stream: Union[PathLike, str, TextIO],
    base_dir: Union[PathLike, AnyStr] = None,
    interpret_as_Clipppy=False,
    force_templating=True,
    scope: Union[Mapping[str, Any], FrameType] = None,
    **kwargs): ...
def loads(
    string: str, *,
    base_dir: Union[PathLike, AnyStr] = None,
    interpret_as_Clipppy=False,
    force_templating=True,
    scope: Union[Mapping[str, Any], FrameType] = None,
    **kwargs): ...
def load_config(
    path_or_stream: Union[PathLike, str, TextIO],
    base_dir: Union[PathLike, AnyStr] = None,
    force_templating=True,
    scope: Union[Mapping[str, Any], FrameType] = None,
    **kwargs
) -> Clipppy: ...
