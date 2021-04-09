import ast
import builtins
import inspect
import io
import os
import typing
import typing as tp
from contextlib import contextmanager
from functools import lru_cache, partial, wraps
from importlib import import_module
from inspect import BoundArguments, Parameter, Signature
from itertools import repeat, starmap
from pathlib import Path
from warnings import warn

import numpy as np
import torch
from frozendict import frozendict
from more_itertools import consume, side_effect
from ruamel import yaml as yaml

# noinspection PyUnresolvedReferences
from . import Clipppy, guide, helpers
from .globals import get_global
from .stochastic import InfiniteSampler, Param, Sampler, SemiInfiniteSampler, stochastic
from .templating import TemplateWithDefaults
from .utils import valueiter
from .utils.signatures import get_param_for_name


_T = typing.TypeVar('_T')


class NodeTypeMismatchError(Exception):
    pass


class YAMLConstructor:
    free_signature = Signature((Parameter('args', Parameter.VAR_POSITIONAL),
                                Parameter('kwargs', Parameter.VAR_KEYWORD)))

    type_to_tag: tp.Dict[tp.Type, str] = {}

    strict_node_type = True

    @classmethod
    def infer_single_tag(cls, node: typing.Union[yaml.Node, _T], param: Parameter) -> typing.Union[yaml.Node, _T]:
        if not isinstance(node, yaml.Node):
            return node
        try:
            if param.annotation is param.empty:
                if param.default in (param.empty, None):
                    # noinspection PyTypeChecker
                    return
                else:
                    hint = type(param.default)
            else:
                hint = (param.annotation if inspect.isclass(param.annotation)
                        else tp.get_origin(param.annotation) or None)

            if not inspect.isclass(hint):
                # TODO: Maybe handle Union??
                # noinspection PyTypeChecker
                return

            hint_is_builtin = any(hint.__module__.startswith(mod)
                                  for mod in ('builtins', 'typing', 'collections'))
            hint_is_str = issubclass(hint, str)
            hint_is_callable = issubclass(hint, type) or (not isinstance(hint, type) and issubclass(hint, tp.Callable))
            target_nodetype = (
                    (hint_is_callable or hint_is_str) and yaml.ScalarNode
                    or hint_is_builtin and issubclass(hint, tp.Mapping) and yaml.MappingNode
                    or hint_is_builtin and issubclass(hint, tp.Iterable) and yaml.SequenceNode
                    or yaml.Node
            )

            for n in valueiter(node):
                if not isinstance(n, target_nodetype):
                    raise NodeTypeMismatchError(f'Expected {target_nodetype} for {param}, but got {n}.')
                if (not hint_is_str
                        and (not hint_is_builtin or target_nodetype in (yaml.ScalarNode, yaml.Node))
                        and n.tag.startswith('tag:yaml.org')):
                    if hint_is_callable:
                        n.tag = f'!py:{n.value}'
                        n.value = ''
                    else:
                        n.tag = cls.type_to_tag.get(hint, f'!py:{hint.__module__}.{hint.__name__}')
        except Exception as e:
            if cls.strict_node_type and isinstance(e, NodeTypeMismatchError):
                raise
            warn(str(e), RuntimeWarning)
        finally:
            return node

    @classmethod
    def infer_tags_from_signature(cls, signature: BoundArguments):
        consume(
            cls.infer_single_tag(node, param)
            for name, val in signature.arguments.items()
            for param in [signature.signature.parameters[name]]
            for node in (val.values() if param.kind == param.VAR_KEYWORD else
                         val if param.kind == param.VAR_POSITIONAL else (val,))
        )

    _args_key = '<'
    _kwargs_key = '<<'

    @classmethod
    def get_args_kwargs(cls, loader: yaml.Loader, node: yaml.Node,
                        signature: Signature = free_signature) -> tp.Optional[BoundArguments]:
        construct_object = partial(loader.construct_object, deep=True)
        if isinstance(node, yaml.ScalarNode):
            try:
                # This sometimes fails because of ruamel/yaml/resolver.py line 370
                node.tag = loader.resolver.resolve(yaml.ScalarNode, node.value, [True, False])
            except IndexError:
                node.tag = loader.DEFAULT_SCALAR_TAG
            val = loader.yaml_constructors[node.tag](loader, node)
            return None if val is None else signature.bind(val)
        elif isinstance(node, yaml.SequenceNode):
            cls.infer_tags_from_signature(signature.bind(*node.value))
            return signature.bind(*map(construct_object, node.value))
        elif isinstance(node, yaml.MappingNode):
            pos_params = (p for param in signature.parameters.values()
                          for p in ((param,) if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
                                    else repeat(param) if param.kind == param.VAR_POSITIONAL else ()))
            args, kwargs = [],  {}
            for key, val in node.value:  # type: typing.Union[yaml.Node, str], typing.Union[yaml.Node, typing.Iterable]
                if key.tag == 'tag:yaml.org,2002:merge':
                    key.tag = loader.DEFAULT_SCALAR_TAG
                key = construct_object(key)
                if key == '__args':
                    key = cls._args_key
                    warn(f'Using "__args" for parameter expansion is deprecated and will soon be considered an ordinary keyword argument.'
                         f' Consider using "{cls._args_key}" instead.', FutureWarning)
                if key == cls._args_key:
                    is_default_seq = val.tag == loader.DEFAULT_SEQUENCE_TAG
                    if is_default_seq:
                        consume(starmap(cls.infer_single_tag, zip(val.value, pos_params)))
                    val = construct_object(val)
                    args.extend(val if is_default_seq else side_effect(partial(next, pos_params), val))
                elif key == cls._kwargs_key:
                    if val.tag in (loader.DEFAULT_SCALAR_TAG, loader.DEFAULT_SEQUENCE_TAG):
                        raise NodeTypeMismatchError(f'Expected {yaml.MappingNode} for "{key}", but got {val}.')
                    elif val.tag == loader.DEFAULT_MAPPING_TAG:
                        consume(cls.infer_single_tag(v, get_param_for_name(signature, construct_object(k))) for k, v in val.value)
                    kwargs.update(construct_object(val))
                else:
                    kwargs[key] = construct_object(cls.infer_single_tag(val, get_param_for_name(signature, key)))

            return signature.bind(*args, **kwargs)
        else:
            raise ValueError(f'Invalid node type: {node}')

    @classmethod
    def construct(cls, obj, loader: yaml.Loader, node: yaml.Node, **kw):
        try:
            signature = inspect.signature(obj, follow_wrapped=False)
        except (TypeError, ValueError):
            signature = cls.free_signature

        signature = cls.get_args_kwargs(loader, node, signature)
        if signature is None:
            return obj
        else:
            signature.apply_defaults()
            signature.arguments.update(kw)
            return obj(*signature.args, **signature.kwargs)

    @classmethod
    def apply(cls, obj):
        return partial(cls.construct, obj)


class PrefixedYAMLConstructor(YAMLConstructor):
    @staticmethod
    def get_class(name: str):
        if '.' in name:
            mods = name.split('.')
            name = mods.pop()

            ret = get_global(mods[0])
            if ret is not None:
                for r in mods[1:]:
                    ret = getattr(ret, r)
            else:
                ret = import_module('.'.join(mods))

            ret = getattr(ret, name)
        else:
            ret = get_global(name)
            if ret is None:
                raise NameError(f'name \'{name}\' is not defined')

        return ret

    @classmethod
    def construct(cls, loader: yaml.Loader, suffix: str, node: yaml.Node, **kwargs):
        return super().construct(cls.get_class(suffix), loader, node, **kwargs)


class PrefixedTensorYAMLConstructor(YAMLConstructor):
    @classmethod
    def construct(cls, loader: yaml.Loader, suffix: str, node: yaml.Node, **kwargs):
        if suffix == 'default':
            kwargs = {'dtype': torch.get_default_dtype(),
                      'device': torch._C._get_default_device(),
                      **kwargs}
        else:
            kwargs = {'dtype': getattr(torch, suffix), **kwargs}
        if not isinstance(kwargs['dtype'], torch.dtype):
            raise ValueError(f'In tag "!torch:{suffix}", "{suffix}" is not a valid torch.dtype.')

        return super().construct(torch.tensor, loader, node, **kwargs)


class PrefixedStochasticYAMLConstructor(YAMLConstructor):
    @classmethod
    def construct(cls, loader: yaml.Loader, suffix: str, node: yaml.Node, **kwargs):
        return super().construct(stochastic, loader, node, name=suffix)


# sys.version_info >= (3, 8)
# spec: tp.Union[str, tp.TypedDict('', **{'from': str, 'import': tp.Union[str, tp.Sequence[str]]}, total=False)]
def _import(*specs: tp.Union[str, tp.Dict]):
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
                                for name in getattr(mod, '__all__', [key for key in vars(mod) if not key.startswith('_')]):
                                    res[name] = getattr(mod, name)
                            else:
                                res[names.asname if names.asname is not None else names.name] = getattr(mod, names.name)
            except SyntaxError as e:
                if not spec.startswith('import'):
                    spec = 'import ' + spec
                    continue  # retry
                else:
                    raise e
            else:
                break
    vars(builtins).update(res)


@contextmanager
def cwd(newcwd: os.PathLike):
    curcwd = Path.cwd()
    try:
        os.chdir(newcwd or '.')
        yield
    finally:
        os.chdir(curcwd)


class ClipppyYAML(yaml.YAML):
    @lru_cache(typed=True)
    def _load_file(self, loader: tp.Callable, *args, **kwargs):
        return loader(*args, **kwargs)

    @staticmethod
    def eval(loader, node: yaml.Node):
        return eval(node.value)

    def npy(self, loader, node: yaml.Node) -> np.ndarray:
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

    def load(self, path_or_stream: tp.Union[os.PathLike, str, tp.TextIO],
             force_templating=True, **kwargs):
        is_a_stream = isinstance(path_or_stream, io.IOBase)
        path = Path((is_a_stream and getattr(path_or_stream, 'name', Path() / 'dummy')) or path_or_stream)
        stream = is_a_stream and path_or_stream or path.open('r')

        if force_templating or kwargs:
            stream = io.StringIO(TemplateWithDefaults(stream.read()).safe_substitute(**kwargs))

        with cwd(self.base_dir or path.parent):
            return super().load(stream)

    def __init__(self, base_dir: tp.Union[os.PathLike, tp.AnyStr] = None, interpret_as_Clipppy=True):
        self.base_dir = base_dir if base_dir is not None else None

        super().__init__(typ='unsafe')
        c: yaml.Constructor = self.constructor
        c.add_constructor('!eval', self.eval)
        c.add_constructor('!npy', self.npy)
        c.add_constructor('!npz', YAMLConstructor.apply(self.npz))
        c.add_constructor('!txt', YAMLConstructor.apply(self.txt))
        c.add_constructor('!pt', YAMLConstructor.apply(self.pt))

        c.add_constructor('!import', YAMLConstructor.apply(_import))
        c.add_multi_constructor('!py:', PrefixedYAMLConstructor.construct)

        c.add_constructor('!tensor', YAMLConstructor.apply(torch.tensor))
        c.add_multi_constructor('!tensor:', PrefixedTensorYAMLConstructor.construct)
        # TODO: Needs to be handled better?
        YAMLConstructor.type_to_tag[torch.Tensor] = '!tensor'

        c.add_constructor('!Stochastic', YAMLConstructor.apply(stochastic))
        c.add_multi_constructor('!Stochastic:', PrefixedStochasticYAMLConstructor.construct)

        c.add_constructor('!Param', YAMLConstructor.apply(Param))
        c.add_constructor('!Sampler', YAMLConstructor.apply(Sampler))
        c.add_constructor('!InfiniteSampler', YAMLConstructor.apply(InfiniteSampler))
        c.add_constructor('!SemiInfiniteSampler', YAMLConstructor.apply(SemiInfiniteSampler))

        if interpret_as_Clipppy:
            r: yaml.Resolver = self.resolver
            r.add_path_resolver('!py:Clipppy', [])
            r.add_path_resolver('!py:Guide', ['guide'])


def __getattr__(name):
    if name == 'MyYAML':
        warn('"MyYAML" was renamed to "ClipppyYAML" and will soon be unavailable.', FutureWarning)
        return ClipppyYAML
