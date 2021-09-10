from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ._clipppy import *


__version__ = '0.42.0a1'
__all__ = 'load_config', 'loads', 'Clipppy', 'ClipppyYAML', '__version__'


def __getattr__(name):
    if name in __all__:
        from . import _clipppy
        globals().update(vars(_clipppy))
        return globals()[name]
    raise AttributeError(f'module {__name__} has no attribute {name}')


def __dir__():
    return __all__
