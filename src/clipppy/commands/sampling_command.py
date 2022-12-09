from __future__ import annotations

from abc import ABC
from contextlib import nullcontext

import pyro

from ..commands import Command
from ..utils.messengers import init_msgr


class SamplingCommand(Command, ABC):
    savename: str = None
    """Filename to save the sample to (or `None` to skip saving)."""

    conditioning: bool = None
    """Whether to retain any conditioning already applied to the model.
       If `False`, `pyro.poutine.handlers.uncondition` will be applied to
       ``model`` before evaluating."""

    @property
    def uncondition(self):
        return pyro.poutine.uncondition() if self.conditioning is False else nullcontext()

    initting: bool = True
    """Whether to respect ``init`` values in config sites."""

    @property
    def init(self):
        return init_msgr if self.initting else nullcontext()
