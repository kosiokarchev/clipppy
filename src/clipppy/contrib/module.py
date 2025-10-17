from contextlib import ExitStack
from typing import ClassVar, Collection, Optional
from typing import Mapping

import attr
import pyro
from pyro.distributions import TorchDistribution
from pyro.nn import PyroModule, PyroSample

from phytorch.utils._typing import _t
from phytorchx import to_tensor
from phytorchx.attrs import AttrsModule


@attr.s(eq=False, auto_attribs=True)
class ClipppyModule(AttrsModule, PyroModule):
    _priors: ClassVar[Mapping[str, TorchDistribution]] = {}
    _inits: ClassVar[Mapping[str, _t]] = {}

    @property
    def own_global_vars(self) -> Collection[str]:
        return self._priors.keys()

    @property
    def global_vars(self):
        yield from self.own_global_vars
        for key, mod in self.named_children():
            if isinstance(mod, ClipppyModule):
                yield from mod.global_vars

    @property
    def own_local_vars(self) -> Collection[str]:
        return ()

    @property
    def local_vars(self):
        yield from self.own_local_vars
        for key, mod in self.named_children():
            if isinstance(mod, ClipppyModule):
                yield from mod.local_vars

    def find_var(self, name) -> Optional[str]:
        for mod in self.modules():
            if isinstance(mod, ClipppyModule) and (name in mod.own_global_vars or name in mod.own_local_vars):
                return mod.get_fullname(name)
        return name

    def __attrs_post_init__(self):
        for name, prior in self._priors.items():
            setattr(self, name, PyroSample(prior))

    def set(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
        return self

    def get_fullname(self, name):
        with self._pyro_context:
            return self._pyro_get_fullname(name)

    @property
    def init(self):
        ctx = ExitStack()
        ctx.enter_context(pyro.condition(data={
            self.get_fullname(key): to_tensor(val) for key, val in self._inits.items()
        }))
        for name, submod in self.named_children():
            if isinstance(submod, ClipppyModule):
                ctx.enter_context(submod.init)
        return ctx
