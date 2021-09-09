from __future__ import annotations

from warnings import warn

from .infinite import *
from .sampler import *
from .stochastic import *


__all__ = stochastic.__all__ + sampler.__all__ + infinite.__all__


def __getattr__(name):
    if name in ('StochasticWrapper', 'stochastic'):
        warn(f'Use \'Stochastic\' instead of \'{name}\', which will soon be unavailable', FutureWarning)
        return Stochastic
    raise AttributeError(name)
