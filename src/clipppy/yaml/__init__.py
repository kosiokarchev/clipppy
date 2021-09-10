from __future__ import annotations

import inspect
import io
import os
from collections import ChainMap
from contextlib import contextmanager
from functools import lru_cache, partial, wraps
from pathlib import Path
from types import FrameType
from typing import Any, AnyStr, Callable, Mapping, TextIO, Union
from warnings import warn

import numpy as np
import torch
from ruamel.yaml import Node, YAML

from .prefixed import stochastic_prefix, tensor_prefix
# TODO: .resolver comes before .constructor!
from .resolver import ClipppyResolver, ImplicitClipppyResolver
from .constructor import ClipppyConstructor as CC
from ..stochastic.infinite import InfiniteUniform, SemiInfiniteUniform
from ..stochastic.sampler import Param, Sampler
from ..stochastic.stochastic import Stochastic
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


def determine_scope(scope: Union[Mapping[str, Any], FrameType] = None):
    if isinstance(scope, Mapping):
        return scope
    if scope is None:
        scope = inspect.stack()[2].frame
    return ChainMap(scope.f_locals, scope.f_globals, scope.f_builtins)


class ClipppyYAML(YAML):
    @lru_cache(typed=True)
    def _load_file(self, loader: Callable, *args, **kwargs):
        return loader(*args, **kwargs)

    @staticmethod
    def eval(loader: CC, node: Node):
        return eval(node.value, {}, loader.scope)

    @wraps(staticmethod(np.loadtxt), assigned=('__annotations__',), updated=())
    def txt(self, *args, **kwargs):
        return self._load_file(np.loadtxt, *args, **kwargs)

    @wraps(staticmethod(np.load), assigned=('__annotations__',), updated=())
    def npy(self, *args, **kwargs):
        return self._load_file(np.load, *args, **kwargs)

    def npz(self, fname: str, key: str = None, **kwargs) -> np.ndarray:
        data = self._load_file(np.load, fname, **kwargs)
        return data if key is None else data[key]

    def pt(self, fname: str, key: str = None, **kwargs):
        data = self._load_file(torch.load, fname, **kwargs)
        return data if key is None else data[key]

    def load(self, path_or_stream: Union[os.PathLike, str, TextIO], force_templating=True,
             scope: Union[Mapping[str, Any], FrameType] = None, **kwargs):
        is_a_stream = isinstance(path_or_stream, io.IOBase)
        path = Path((is_a_stream and getattr(path_or_stream, 'name', Path() / 'dummy')) or path_or_stream)
        stream = is_a_stream and path_or_stream or path.open('r')

        if force_templating or kwargs:
            stream = io.StringIO(TemplateWithDefaults(stream.read()).safe_substitute(**kwargs))

        self.constructor.scope = determine_scope(scope)

        with cwd(self.base_dir or path.parent):
            return super().load(stream)

    resolver: ClipppyResolver
    constructor: CC

    def __init__(self, base_dir: Union[os.PathLike, AnyStr] = None, interpret_as_Clipppy=True):
        self.base_dir = base_dir if base_dir is not None else None

        super().__init__(typ='unsafe', pure=True)

        self.Resolver = ImplicitClipppyResolver if interpret_as_Clipppy else ClipppyResolver
        self.Constructor = CC


CC.add_constructor('!import', CC.apply_bound(CC.import_))
CC.add_multi_constructor('!py:', CC.apply_bound_prefixed(CC.resolve_name))

CC.add_constructor('!eval', ClipppyYAML.eval)

for func in (ClipppyYAML.txt, ClipppyYAML.npy, ClipppyYAML.npz, ClipppyYAML.pt):
    CC.add_constructor(f'!{func.__name__}', CC.apply_bound(func, _cls=ClipppyYAML))

CC.add_constructor('!tensor', CC.apply(torch.tensor))
CC.add_multi_constructor('!tensor:', CC.apply_prefixed(tensor_prefix))
# TODO: Needs to be handled better?
CC.type_to_tag[torch.Tensor] = '!tensor'

for typ in (Stochastic, Param, Sampler):
    CC.add_constructor(f'!{typ.__name__}', CC.apply(typ))
CC.add_constructor('!InfiniteSampler', CC.apply(partial(Sampler, d=InfiniteUniform())))
CC.add_constructor('!SemiInfiniteSampler', CC.apply(partial(Sampler, d=SemiInfiniteUniform())))
CC.add_multi_constructor('!Stochastic:', CC.apply_prefixed(stochastic_prefix))


def _register_globals():
    from . import hooks
    from .. import clipppy, stochastic, guide, helpers
    for mod in (clipppy, stochastic, guide, helpers):
        CC.builtins.update(**{a: getattr(mod, a) for a in mod.__all__})
    CC.builtins.update({'torch': torch, 'np': np, 'numpy': np})


_register_globals()
del _register_globals


def __getattr__(name):
    if name == 'MyYAML':
        warn('\'MyYAML\' was renamed to \'ClipppyYAML\' and will soon be unavailable.', FutureWarning)
        return ClipppyYAML
    raise AttributeError(f'module {__name__} has no attribute {name}')
