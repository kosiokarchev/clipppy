from __future__ import annotations

from itertools import starmap
from typing import Any, MutableMapping

import torch
from more_itertools import consume

from ..stochastic.stochastic import Stochastic


def tensor_prefix(suffix: str, kwargs: MutableMapping[str, Any]):
    consume(starmap(kwargs.setdefault, (
        (('dtype', torch.get_default_dtype()), ('device', torch._C._get_default_device()))
        if suffix == 'default' else (('dtype', getattr(torch, suffix, None)),)
    )))
    if not isinstance(kwargs['dtype'], torch.dtype):
        raise ValueError(f'In tag \'!torch:{suffix}\', \'{suffix}\' is not a valid torch.dtype.')

    return torch.tensor, kwargs


def stochastic_prefix(suffix: str, kwargs: MutableMapping[str, Any]):
    kwargs.setdefault('name', suffix)
    return Stochastic, kwargs
