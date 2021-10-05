from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional

import pyro
import torch
from frozendict import frozendict
from pyro import poutine
from pyro.infer.autoguide.guides import prototype_hide_fn
from pyro.nn import PyroModule
from torch import Tensor

from .group_spec import GroupSpec
from .sampling_group import SamplingGroup
from ..utils.pyro import AbstractPyroModuleMeta, init_msgr
from ..utils.typing import _Site


class BaseGuide(PyroModule, metaclass=AbstractPyroModuleMeta):
    """"""
    def __init__(self, model=None, name=''):
        super().__init__(name)

        self.model: Callable = model
        self.prototype_trace: Optional[pyro.poutine.Trace] = None
        self.is_setup = False

    def _setup_prototype(self, *args, **kwargs):
        # TODO: Python 3.10: Parenthesized context managers
        with poutine.trace() as trace,\
             poutine.block(hide_fn=prototype_hide_fn),\
             init_msgr:
            self.model(*args, **kwargs)
        self.prototype_trace = trace.trace

    @staticmethod
    def _set_init(site: _Site, init: Mapping[str, Tensor]):
        if site['name'] in init:
            site['value'] = init[site['name']]
        return site['infer']

    def setup(self, *args, init: Mapping[str, Tensor] = frozendict(), **kwargs) -> MutableMapping[str, Any]:
        old_children = dict(self.named_children())
        for child in old_children:
            delattr(self, child)

        # TODO: custom InitMessenger class
        with pyro.poutine.infer_config(config_fn=partial(self._set_init, init=init)):
            self._setup_prototype(*args, **kwargs)
        self.is_setup = True
        return old_children

    @abstractmethod
    def guide(self, *args, **kwargs) -> MutableMapping[str, torch.Tensor]: ...

    def forward(self, *args, **kwargs):
        if not self.is_setup:
            self.setup(*args, **kwargs)
        result = self.guide(*args, **kwargs)
        return result

    def __getstate__(self):
        # Don't pickle the model
        state = self.__dict__.copy()
        state['model'] = None
        return state


class Guide(BaseGuide):
    children: Callable[[], Iterable[SamplingGroup]]

    add_noise = {}  # for backward compatibility

    def __init__(self, *specs: GroupSpec, model=None, name='', add_noise=None):
        super().__init__(model=model, name=name)

        self.specs: Iterable[GroupSpec] = specs
        self.add_noise = add_noise or {}

    def setup(self, *args, init: Mapping[str, Tensor] = frozendict(), **kwargs) -> MutableMapping[str, SamplingGroup]:
        old_children = super().setup(*args, init=init, **kwargs)

        sites = [site for name, site in self.prototype_trace.iter_stochastic_nodes()]
        for spec in self.specs:
            group = spec.make_group(sites)
            if group:
                for site in group.sites.values():
                    sites.remove(site)
                setattr(self, spec.name, group)
        # if sites:
        #     setattr(self, 'Default', GroupSpec().make_group(sites))

        return old_children

    def guide(self, *args, **kwargs) -> MutableMapping[str, torch.Tensor]:
        # Union of all model samples dicts from self.children
        return dict((key, val + torch.normal(0., self.add_noise[key], val.shape)
                     if key in self.add_noise else val)
                    for group in self.children() if group.active
                    for key, val in group()[1].items())
