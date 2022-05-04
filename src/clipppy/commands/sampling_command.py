from __future__ import annotations

from abc import ABC
from contextlib import nullcontext

import pyro

from clipppy.commands import Command
from clipppy.utils.pyro import init_msgr


class SamplingCommand(Command, ABC):
    savename: str = None
    """Filename to save the sample to (or `None` to skip saving)."""

    conditioning: bool = False
    """Whether to retain any conditioning already applied to the model.
       If a false value, `pyro.poutine.handlers.uncondition` will
       be applied to ``model`` before evaluating."""

    @property
    def uncondition(self):
        return pyro.poutine.uncondition() if not self.conditioning else nullcontext()

    initting: bool = True
    """Whether to respect ``init`` values in config sites."""

    @property
    def init(self):
        return init_msgr if self.initting else nullcontext()
