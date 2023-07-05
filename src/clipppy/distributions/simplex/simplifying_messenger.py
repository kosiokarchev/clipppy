from __future__ import annotations

from typing import Mapping, Iterable, Union, MutableMapping

import pyro
from more_itertools import always_iterable
from pyro.distributions import Delta
from pyro.poutine.messenger import Messenger
from torch import Tensor

from ...utils.typing import _Distribution, _Site


class SimplifyingMessenger(Messenger):
    def __init__(self, dists: Mapping[Union[str, Iterable[str]], _Distribution]):
        super().__init__()
        self.dists: Mapping[str, tuple[tuple[str], _Distribution]] = {
            key: (keys, val)
            for keys, val in dists.items() for keys in [tuple(always_iterable(keys))]
            for key in keys
        }
        self.values: MutableMapping[str, Tensor] = {}

    def __enter__(self):
        self.values.clear()
        return super().__enter__()

    @staticmethod
    def get_name(names: tuple[str]):
        return '_simplification_' + '_&_'.join(names)

    def _pyro_sample(self, msg: _Site):
        if (name := msg['name']) in self.dists:
            if name not in self.values:
                names, dist = self.dists[name]
                sample = pyro.sample(self.get_name(names), dist, msg['args'])
                self.values.update(zip(names, sample.unbind(-1)))

            # -> deterministic
            msg['value'] = self.values[name]
            msg['fn'] = Delta(msg['value'], event_dim=self.dists[name][1].event_dim-1).mask(False)
            msg['infer']['_deterministic'] = True

    def __repr__(self):
        return f'{type(self).__name__}{tuple(self.dists.keys())}'
