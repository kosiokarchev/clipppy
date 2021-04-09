import inspect
import io
import os
from collections import ChainMap
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
from types import FrameType
from typing import Any, AnyStr, Callable, ClassVar, Mapping, MutableMapping, TextIO, Union
from warnings import warn

import numpy as np
import torch
from ruamel.yaml import Constructor, Node, Resolver, YAML

from .constructor import YAMLConstructor
from .prefixed import PrefixedStochasticYAMLConstructor, PrefixedTensorYAMLConstructor
from .py import PyYAMLConstructor
from .. import clipppy, guide, helpers, stochastic
from ..stochastic import InfiniteSampler, Param, Sampler, SemiInfiniteSampler, stochastic as Stochastic
from ..templating import TemplateWithDefaults


__all__ = 'ClipppyYAML',


@contextmanager
def cwd(newcwd: os.PathLike):
    curcwd = Path.cwd()
    try:
        os.chdir(newcwd or '.')
        yield
    finally:
        os.chdir(curcwd)


def frame_namespace(frame: Union[FrameType, int]):
    if isinstance(frame, int):
        frame = inspect.stack()[frame+1].frame
    return ChainMap(frame.f_locals, frame.f_globals, frame.f_builtins)


class ClipppyYAML(YAML):
    builtins: ClassVar[Mapping[str, Any]] = ChainMap({}, globals(), __builtins__)
    _scope: MutableMapping[str, Any] = None

    @property
    def scope(self):
        if self._scope is None:
            self.scope = {}
        return self._scope

    @scope.setter
    def scope(self, scope: Mapping[str, Any]):
        self._scope = ChainMap(scope, self.builtins)

    @lru_cache(typed=True)
    def _load_file(self, loader: Callable, *args, **kwargs):
        return loader(*args, **kwargs)

    def eval(self, loader, node: Node):
        return eval(node.value, {}, self.scope)

    def npy(self, loader, node: Node) -> np.ndarray:
        return self._load_file(np.load, node.value)

    def npz(self, fname: str, key: str = None) -> np.ndarray:
        data = self._load_file(np.load, fname)
        return data if key is None else data[key]

    @wraps(np.loadtxt)
    def txt(self, *args, **kwargs):
        return self._load_file(np.loadtxt, *args, **kwargs)

    def pt(self, fname: str, key: str = None, **kwargs):
        data = self._load_file(torch.load, fname, **kwargs)
        return data if key is None else data[key]

    def load(self, path_or_stream: Union[os.PathLike, str, TextIO], force_templating=True,
             scope: Union[Mapping[str, Any], FrameType, int] = 0, **kwargs):
        is_a_stream = isinstance(path_or_stream, io.IOBase)
        path = Path((is_a_stream and getattr(path_or_stream, 'name', Path() / 'dummy')) or path_or_stream)
        stream = is_a_stream and path_or_stream or path.open('r')

        if force_templating or kwargs:
            stream = io.StringIO(TemplateWithDefaults(stream.read()).safe_substitute(**kwargs))

        self.scope = (scope if isinstance(scope, Mapping) else
                      frame_namespace(scope+1 if isinstance(scope, int) else scope))

        with cwd(self.base_dir or path.parent):
            return super().load(stream)

    def __init__(self, base_dir: Union[os.PathLike, AnyStr] = None, interpret_as_Clipppy=True):
        self.base_dir = base_dir if base_dir is not None else None

        super().__init__(typ='unsafe')
        c: Constructor = self.constructor

        self.py_constructor = PyYAMLConstructor(self)
        c.add_multi_constructor('!py:', self.py_constructor.construct)
        c.add_constructor('!import', YAMLConstructor.apply(self.py_constructor.import_))

        c.add_constructor('!eval', self.eval)
        c.add_constructor('!npy', self.npy)
        c.add_constructor('!npz', YAMLConstructor.apply(self.npz))
        c.add_constructor('!txt', YAMLConstructor.apply(self.txt))
        c.add_constructor('!pt', YAMLConstructor.apply(self.pt))

        c.add_constructor('!tensor', YAMLConstructor.apply(torch.tensor))
        c.add_multi_constructor('!tensor:', PrefixedTensorYAMLConstructor.construct)
        # TODO: Needs to be handled better?
        YAMLConstructor.type_to_tag[torch.Tensor] = '!tensor'

        c.add_constructor('!Stochastic', YAMLConstructor.apply(Stochastic))
        c.add_multi_constructor('!Stochastic:', PrefixedStochasticYAMLConstructor.construct)

        c.add_constructor('!Param', YAMLConstructor.apply(Param))
        c.add_constructor('!Sampler', YAMLConstructor.apply(Sampler))
        c.add_constructor('!InfiniteSampler', YAMLConstructor.apply(InfiniteSampler))
        c.add_constructor('!SemiInfiniteSampler', YAMLConstructor.apply(SemiInfiniteSampler))

        if interpret_as_Clipppy:
            r: Resolver = self.resolver
            r.add_path_resolver(f'!py:{clipppy.Clipppy.__name__}', [])
            r.add_path_resolver(f'!py:{guide.guide.Guide.__name__}', ['guide'])


def register_globals(**kwargs):
    ClipppyYAML.builtins.update(kwargs)


for mod in (clipppy, stochastic, guide, helpers):
    register_globals(**{a: getattr(mod, a) for a in mod.__all__})


def __getattr__(name):
    if name == 'MyYAML':
        warn('\'MyYAML\' was renamed to \'ClipppyYAML\' and will soon be unavailable.', FutureWarning)
        return ClipppyYAML
