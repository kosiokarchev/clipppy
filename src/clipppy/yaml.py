import builtins
import re
import sys
import typing as tp
from functools import partial
from importlib import import_module

import numpy as np
import torch
from ruamel import yaml as yaml

from .globals import get_global
from .stochastic import stochastic, Sampler


class YAMLConstructor:
    @staticmethod
    def get_args_kwargs(loader, node) -> tp.Optional[tp.Tuple[tp.List, tp.Dict]]:
        if isinstance(node, yaml.ScalarNode):
            node.tag = loader.resolve(yaml.ScalarNode, node.value, [True, False])
            val = loader.construct_object(node)
            if val is None:
                val = loader.yaml_constructors[node.tag](loader, node)
            return None if val is None else ([], {})
        if isinstance(node, yaml.SequenceNode):
            return [loader.construct_object(n, deep=True)
                    for n in node.value], {}
        elif isinstance(node, yaml.MappingNode):
            kwargs = {loader.construct_object(key, deep=True): loader.construct_object(val, deep=True)
                      for key, val in node.value}
            return kwargs.pop('__args', ()), kwargs

    @classmethod
    def construct(cls, obj, loader: yaml.Loader, node: yaml.Node, **kw):
        argskwargs = cls.get_args_kwargs(loader, node)
        if argskwargs is None:
            return obj
        else:
            args, kwargs = argskwargs
            kwargs.update(kw)
            return obj(*args, **kwargs)

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
        dtype = getattr(torch, suffix)
        if not isinstance(getattr(torch, suffix), torch.dtype):
            raise ValueError(f'In tag "!torch:{suffix}", "{suffix}" is not a valid torch.dtype.')

        return super().construct(torch.tensor, loader, node, dtype=dtype)


class PrefixedStochasticYAMLConstructor(YAMLConstructor):
    @classmethod
    def construct(cls, loader: yaml.Loader, suffix: str, node: yaml.Node, **kwargs):
        return super().construct(stochastic, loader, node, name=suffix)


# spec: tp.Union[str, tp.TypedDict('', **{'from': str, 'import': tp.Union[str, tp.Sequence[str]]}, total=False)]
def _import(*specs: tp.Union[str, tp.Dict]):
    for spec in specs:
        if isinstance(spec, str):
            module = __import__(spec)
            setattr(builtins, module.__name__, module)
        else:
            fromlist = spec.get('import', [])
            if isinstance(fromlist, str):
                fromlist = [re.split(r'\s*as\s*', name)
                            for name in re.split(r'\s*,\s*', fromlist)]

            module = import_module(spec['from'])

            if not fromlist:
                fromlist = getattr(module, '__all__', [key for key in vars(module) if not key.startswith('_')])

            fromlist = [((name,)*2 if isinstance(name, str)
                         else name*2 if len(name)==1 else name)
                        for name in fromlist]

            vars(builtins).update({oname: getattr(module, iname) for iname, oname in fromlist})


class MyYAML(yaml.YAML):
    @staticmethod
    def eval(loader, node: yaml.Node):
        return eval(node.value)

    @staticmethod
    def npy(loader, node: yaml.Node) -> np.ndarray:
        return np.load(node.value)

    @staticmethod
    def npz(fname: str, key: str = None) -> np.ndarray:
        data = np.load(fname)
        return data if key is None else data[key]

    @staticmethod
    def pt(fname: str, key: str = None):
        data = torch.load(fname)
        return data if key is None else data[key]

    def __init__(self):
        super().__init__(typ='unsafe')
        c: yaml.Constructor = self.constructor
        c.add_constructor('!eval', self.eval)
        c.add_constructor('!npy', self.npy)
        c.add_constructor('!npz', YAMLConstructor.apply(self.npz))
        c.add_constructor('!pt', YAMLConstructor.apply(self.pt))

        c.add_constructor('!import', YAMLConstructor.apply(_import))
        c.add_multi_constructor('!py:', PrefixedYAMLConstructor.construct)

        c.add_constructor('!tensor', YAMLConstructor.apply(torch.tensor))
        c.add_multi_constructor('!tensor:', PrefixedTensorYAMLConstructor.construct)

        c.add_constructor('!Stochastic', YAMLConstructor.apply(stochastic))
        c.add_multi_constructor('!Stochastic:', PrefixedStochasticYAMLConstructor.construct)

        c.add_constructor('!Sampler', YAMLConstructor.apply(Sampler))

        r: yaml.Resolver = self.resolver
        r.add_path_resolver('!py:Clipppy', ['clipppy'])
        r.add_path_resolver('!py:Guide', ['clipppy', 'guide'])
