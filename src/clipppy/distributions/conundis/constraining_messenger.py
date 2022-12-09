from __future__ import annotations

from typing import Mapping

from pyro.poutine.messenger import Messenger

from .conundis_mixin import _constraintT, ConUnDisMixin
from ...utils.typing import _Site


class ConstrainingMessenger(Messenger):
    def __init__(self, ranges: Mapping[str, tuple[_constraintT, _constraintT]]):
        super().__init__()
        self.ranges = ranges

    def _pyro_sample(self, msg: _Site):
        if (name := msg['name']) in self.ranges:
            msg['fn'] = ConUnDisMixin.new_constrained(msg['fn'], *self.ranges[name])
