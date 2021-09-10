from __future__ import annotations

import io
import os
from types import FrameType
from typing import Any, AnyStr, Mapping, TextIO, Union

from .clipppy import Clipppy
from .yaml import ClipppyYAML, determine_scope


__all__ = 'load', 'loads', 'load_config', 'Clipppy', 'ClipppyYAML'


def load(path_or_stream: Union[os.PathLike, str, TextIO],
         base_dir: Union[os.PathLike, AnyStr] = None,
         interpret_as_Clipppy=False, force_templating=True,
         scope: Union[Mapping[str, Any], FrameType] = None,
         **kwargs):
    return (ClipppyYAML(base_dir=base_dir, interpret_as_Clipppy=interpret_as_Clipppy)
            .load(path_or_stream, force_templating=force_templating,
                  scope=determine_scope(scope), **kwargs))


def loads(string: str, **kwargs):
    return load(io.StringIO(string), scope=determine_scope(kwargs.pop('scope', None)), **kwargs)


def load_config(*args, **kwargs) -> Clipppy:
    return load(*args, scope=determine_scope(kwargs.pop('scope', None)), interpret_as_clipppy=True, **kwargs)
