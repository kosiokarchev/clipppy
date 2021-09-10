from __future__ import annotations

from typing import Any, Callable, Iterable, MutableMapping, Optional

import pyro
import torch
from pyro import poutine
from pyro.infer.autoguide.guides import prototype_hide_fn
from pyro.nn import PyroModule
from pyro.poutine.indep_messenger import CondIndepStackFrame

from .group_spec import GroupSpec
from .sampling_group import SamplingGroup
from ..utils.pyro import init_msgr


class BaseGuide(PyroModule):
    """"""
    def __init__(self, model=None, name=''):
        super().__init__(name)

        self.model: Callable = model

        self.prototype_trace: Optional[pyro.poutine.Trace] = None

        self.frames: MutableMapping[str, CondIndepStackFrame] = {}
        self.plates: MutableMapping[str, pyro.plate] = {}

        self.is_setup = False

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        with poutine.block():
            with poutine.trace() as trace:
                with poutine.block(hide_fn=prototype_hide_fn):
                    with init_msgr:
                        self.model(*args, **kwargs)
        self.prototype_trace = trace.trace

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            for frame in site["cond_indep_stack"]:
                if not frame.vectorized:
                    raise NotImplementedError("EasyGuide does not support sequential pyro.plate")
                self.frames[frame.name] = frame

    def setup(self, *args, **kwargs) -> MutableMapping[str, Any]:
        old_children = dict(self.named_children())
        for child in old_children:
            delattr(self, child)

        self.is_setup = True
        self._setup_prototype(*args, **kwargs)
        return old_children

    def plate(self, name, size=None, subsample_size=None, subsample=None, *args, **kwargs):
        if name not in self.plates:
            self.plates[name] = pyro.plate(name, size, subsample_size, subsample, *args, **kwargs)
        return self.plates[name]

    def guide(self, *args, **kwargs) -> MutableMapping[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        if not self.is_setup:
            self.setup(*args, **kwargs)
        result = self.guide(*args, **kwargs)
        self.plates.clear()
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

    def setup(self, *args, **kwargs) -> MutableMapping[str, SamplingGroup]:
        old_children = super().setup(*args, **kwargs)

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
