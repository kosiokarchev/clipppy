from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ._clipppy import *

from ._version import __version__, version, __version_tuple__, version_tuple


__all__ = 'load_config', 'load', 'loads', 'Clipppy', 'ClipppyYAML'


def __getattr__(name):
    if name in __all__:
        from . import _clipppy
        globals().update(vars(_clipppy))
        return globals()[name]
    raise AttributeError(f'module {__name__} has no attribute {name}')


def __dir__():
    return __all__
