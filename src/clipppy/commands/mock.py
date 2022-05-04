from __future__ import annotations

from contextlib import contextmanager, ExitStack
from typing import ContextManager, Sequence

import pyro
import pyro.poutine
import torch

from .sampling_command import SamplingCommand
from ..utils.typing import _Model


@contextmanager
def multi_context(*cms):
    # https://stackoverflow.com/a/45681273/7185647
    with ExitStack() as stack:
        yield list(map(stack.enter_context, cms))


class Mock(SamplingCommand):
    """Generate mock data from the model prior."""

    extra_messengers: Sequence[ContextManager] = ()

    def forward(self, model: _Model, *args, **kwargs) -> pyro.poutine.Trace:
        with pyro.poutine.trace() as trace, multi_context(*self.extra_messengers), self.init, self.plate, self.uncondition:
            model(*args, **kwargs)

        if self.savename:
            torch.save(trace.trace, self.savename)

        return trace.trace


