from __future__ import annotations

import pyro
import pyro.poutine
import torch

from .command import SamplingCommand
from ..utils.pyro import init_msgr
from ..utils.typing import _Model


class Mock(SamplingCommand):
    """Generate mock data from the model prior."""

    def forward(self, model: _Model, *args, **kwargs) -> pyro.poutine.Trace:
        with pyro.poutine.trace() as trace, init_msgr, self.plate, self.uncondition:
            model(*args, **kwargs)

        if self.savename:
            torch.save(trace.trace, self.savename)

        return trace.trace


