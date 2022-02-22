from __future__ import annotations

from itertools import starmap
from typing import Any, Callable, Mapping, MutableMapping, NamedTuple

import torch
from more_itertools import consume

from ..utils import torch_get_default_device


class PrefixedReturn(NamedTuple):
    func: Callable
    kwargs: Mapping[str, Any]


def tensor_prefix(suffix: str, kwargs: MutableMapping[str, Any]):
    consume(starmap(kwargs.setdefault, (
        (('dtype', torch.get_default_dtype()), ('device', torch_get_default_device()))
        if suffix == 'default' else (('dtype', getattr(torch, suffix, None)),)
    )))
    if not isinstance(kwargs['dtype'], torch.dtype):
        raise ValueError(f'In tag \'!torch:{suffix}\', \'{suffix}\' is not a valid torch.dtype.')

    return PrefixedReturn(torch.tensor, kwargs)


def named_prefix(obj, suffix: str, kwargs: MutableMapping[str, Any]):
    kwargs.setdefault('name', suffix)
    return PrefixedReturn(obj, kwargs)
