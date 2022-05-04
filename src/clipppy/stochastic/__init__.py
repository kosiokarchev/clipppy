from __future__ import annotations

from typing import Any, Optional, Union
from warnings import warn

from .infinite import *
from .sampler import *
from .stochastic import *


__all__ = stochastic.__all__ + sampler.__all__ + infinite.__all__ + ('find_sampler',)


def find_sampler(root: Union[NamedSampler, PseudoSampler, Sampler, Stochastic, Any], name: str) -> Optional[NamedSampler]:
    if isinstance(root, NamedSampler) and root.name == name:
        return root
    elif isinstance(root, PseudoSampler):
        return find_sampler(root.func_or_val, name)
    elif isinstance(root, Sampler):
        return find_sampler(root.d, name)
    elif isinstance(root, Stochastic):
        for key, spec in root.stochastic_specs.items():
            if key == name:
                return spec
            elif (ret := find_sampler(spec, name)) is not None:
                return ret


def __getattr__(name):
    if name in ('StochasticWrapper', 'stochastic'):
        warn(f'Use \'Stochastic\' instead of \'{name}\', which will soon be unavailable', FutureWarning)
        return Stochastic
    raise AttributeError(name)
