import ast
import builtins as __builtins__
import inspect
import io
import os
import re
from collections import ChainMap
from contextlib import contextmanager
from functools import lru_cache, wraps
from importlib import import_module
from pathlib import Path
from types import FrameType
from typing import Any, AnyStr, Callable, ClassVar, Dict, Mapping, MutableMapping, TextIO, Union
from warnings import warn

import numpy as np
import torch
from ruamel.yaml import Constructor, Node, Resolver, YAML

from .constructor import YAMLConstructor
from .prefixed import PrefixedStochasticYAMLConstructor, PrefixedTensorYAMLConstructor
from .py import PyYAMLConstructor
from .. import Clipppy
from ..stochastic import stochastic as Stochastic
from ..stochastic.infinite import InfiniteSampler, SemiInfiniteSampler
from ..stochastic.sampler import Param, Sampler
from ..templating import TemplateWithDefaults


__builtins__ = vars(__builtins__)  # ensure this, since the standard does not guarantee it

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

    # sys.version_info >= (3, 8) TODO: true?
    # spec: Union[str, TypedDict('', **{'from': str, 'import': Union[str, Sequence[str]]}, total=False)]
    def import_(self, *specs: Union[str, Dict]):
        res = {}
        for spec in specs:
            if not isinstance(spec, str):
                spec = f'from {spec["from"]} import {spec["import"] if isinstance(spec["import"], str) else ", ".join(spec["import"])}'
            while True:
                try:
                    for stmt in ast.parse(spec).body:
                        if isinstance(stmt, ast.Import):
                            for names in stmt.names:
                                if names.asname is None:
                                    res[names.name] = __import__(names.name)
                                else:
                                    res[names.asname] = import_module(names.name)
                        elif isinstance(stmt, ast.ImportFrom):
                            mod = import_module(stmt.module)
                            for names in stmt.names:
                                if names.name == '*':
                                    for name in getattr(mod, '__all__',
                                                        [key for key in vars(mod) if not key.startswith('_')]):
                                        res[name] = getattr(mod, name)
                                else:
                                    res[names.asname if names.asname is not None else names.name] = getattr(mod, names.name)
                        else:
                            raise SyntaxError('Only import/import from statements are allowed.')
                except SyntaxError as e:
                    if not spec.startswith('import'):
                        spec = 'import ' + spec
                        continue  # retry
                    else:
                        raise e
                else:
                    break
        self.scope.update(res)

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
             scope: Union[Mapping[str, Any], FrameType] = None, **kwargs):
        is_a_stream = isinstance(path_or_stream, io.IOBase)
        path = Path((is_a_stream and getattr(path_or_stream, 'name', Path() / 'dummy')) or path_or_stream)
        stream = is_a_stream and path_or_stream or path.open('r')

        if force_templating or kwargs:
            stream = io.StringIO(TemplateWithDefaults(stream.read()).safe_substitute(**kwargs))

        self.scope = determine_scope(scope)

        with cwd(self.base_dir or path.parent):
            return super().load(stream)

    @classmethod
    def register_globals(cls, **kwargs):
        cls.builtins.update(kwargs)

    def __init__(self, base_dir: Union[os.PathLike, AnyStr] = None, interpret_as_Clipppy=True):
        self.base_dir = base_dir if base_dir is not None else None

        super().__init__(typ='unsafe')
        self.py_constructor = PyYAMLConstructor(self)

        r: Resolver = self.resolver
        r.add_implicit_resolver(self.py_constructor._args_tag, re.compile('<'), '<')

        if interpret_as_Clipppy:
            r.add_path_resolver(f'!py:{Clipppy.__name__}', [])

        c: Constructor = self.constructor
        # TODO: FORCE DEPTH!!
        c.deep_construct = True

        c.add_multi_constructor('!py:', self.py_constructor.construct)
        c.add_constructor('!import', YAMLConstructor.apply(self.import_))

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


def __getattr__(name):
    if name == 'MyYAML':
        warn('\'MyYAML\' was renamed to \'ClipppyYAML\' and will soon be unavailable.', FutureWarning)
        return ClipppyYAML


def _register_globals():
    from .. import clipppy, stochastic, guide, helpers
    for mod in (clipppy, stochastic, guide, helpers):
        ClipppyYAML.register_globals(**{a: getattr(mod, a) for a in mod.__all__})


_register_globals()
del _register_globals
