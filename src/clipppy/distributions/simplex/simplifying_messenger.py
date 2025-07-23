from __future__ import annotations

from typing import Mapping, Iterable, MutableMapping

import pyro
from more_itertools import always_iterable
from pyro.poutine.messenger import Messenger
from torch import Tensor
from typing_extensions import TypeAlias

from ...utils.pyro import make_deterministic
from ...utils.typing import _Distribution, _Site

_KT: TypeAlias = str | Iterable[str]


class SimplifyingMessenger(Messenger):
    def __init__(self, dists: Mapping[_KT, _Distribution]):
        super().__init__()
        self.dists: Mapping[str, tuple[_KT, _Distribution]] = {
            key: (group, val)
            for group, val in dists.items()
            for key in always_iterable(group)
        }
        self.values: MutableMapping[str, Tensor] = {}

    def __enter__(self):
        self.values.clear()
        return super().__enter__()

    @staticmethod
    def get_name(names: _KT):
        return '_simplification_' + '_&_'.join(always_iterable(names))

    def _pyro_sample(self, msg: _Site):
        if (name := msg['name']) in self.dists:
            group, dist = self.dists[name]
            is_multisite = not isinstance(group, str)

            if name not in self.values:
                sample = pyro.sample(self.get_name(group), dist)
                if is_multisite:
                    self.values.update(zip(group, sample.unbind(-1)))
                else:
                    self.values[group] = sample

            make_deterministic(msg, self.values[name], dist.event_dim - is_multisite)

    def __repr__(self):
        return f'{type(self).__name__}{tuple(self.dists.keys())}'
