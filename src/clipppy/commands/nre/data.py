from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import ContextManager, Iterable, Mapping, Union

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from ... import clipppy
from ...distributions.conundis import ConstrainingMessenger


__all__ = 'ClipppyDataset', 'CPDataset'


@dataclass
class ClipppyDataset(IterableDataset[tuple[Mapping[str, Tensor], Mapping[str, Tensor]]]):
    config: clipppy.Clipppy
    param_names: Iterable[str]
    obs_names: Iterable[str]
    batch_size: int = 0
    mock_args: Mapping = field(default_factory=lambda: dict(
        initting=False, conditioning=False))

    @property
    def context(self) -> ContextManager:
        return nullcontext()

    def __next__(self):
        while True:
            try:
                with torch.no_grad(), self.context:
                    mock = self.config.mock(
                        plate_stack=(self.batch_size,) if self.batch_size else None,
                        **self.mock_args
                    )
                return tuple(
                    OrderedDict((name, mock.nodes[name]['value']) for name in names)
                    for names in (self.param_names, self.obs_names)
                )
            except ValueError as e:
                pass

    def __iter__(self):
        return self


@dataclass
class CPDataset(ClipppyDataset):
    ranges: Mapping[str, tuple[Union[float, Tensor, None], Union[float, Tensor, None]]] = None

    @property
    def context(self):
        return ConstrainingMessenger(self.ranges)
