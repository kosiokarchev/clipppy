import builtins
import inspect
import re
import typing as tp
from functools import partial
from importlib import import_module
from inspect import Signature, BoundArguments, Parameter
from warnings import warn

import numpy as np
import torch
from ruamel import yaml as yaml

from .globals import get_global
from .stochastic import stochastic, Sampler


class YAMLConstructor:
    free_signature = Signature((Parameter('args', Parameter.VAR_POSITIONAL),
                                Parameter('kwargs', Parameter.VAR_KEYWORD)))

    type_to_tag: tp.Dict[tp.Type, str] = {}

    @classmethod
    def fix_signature(cls, signature: BoundArguments) -> BoundArguments:
        for name, node in signature.arguments.items():
            try:
                param = signature.signature.parameters[name]
                if param.annotation is param.empty:
                    if param.default in (param.empty, None):
                        continue
                    else:
                        hint = type(param.default)
                else:
                    # TODO: migrate to 3.8:
                    # tp.get_origin(param.annotations)
                    hint = (param.annotation if inspect.isclass(param.annotation)
                            else tp.get_origin(param.annotation) or object)
                if any(hint.__module__.startswith(mod)
                       for mod in ('builtins', 'typing', 'collections')):
                    if hint is tp.Union:
                        pass  # Can't handle unions (for now?)
                    elif issubclass(hint, str):
                        if not isinstance(node, yaml.ScalarNode):
                            warn(f'Expected string node for {param}, but got {node}.')
                    elif issubclass(hint, tp.Mapping):
                        if not isinstance(node, yaml.MappingNode):
                            warn(f'Expected a mapping node for {param}, but got {node}.')
                    elif issubclass(hint, tp.Iterable):
                        if not isinstance(node, yaml.SequenceNode):
                            warn(f'Expected a sequece node for {param}, but got {node}.')
                else:
                    if node.tag.startswith('tag:yaml.org'):
                        node.tag = cls.type_to_tag.get(hint, f'!py:{hint.__module__}.{hint.__name__}')
            except Exception as e:
                warn(e)

        return signature

    @classmethod
    def get_args_kwargs(cls, loader: yaml.Loader, node: yaml.Node,
                        signature: Signature = free_signature) -> tp.Optional[BoundArguments]:
        if isinstance(node, yaml.ScalarNode):
            node.tag = loader.resolve(yaml.ScalarNode, node.value, [True, False])
            val = loader.construct_object(node)
            if val is None:
                val = loader.yaml_constructors[node.tag](loader, node)
            return None if val is None else signature.bind(val)
        elif isinstance(node, yaml.SequenceNode):
            bound = signature.bind(*node.value)
            subnodes = node.value
        elif isinstance(node, yaml.MappingNode):
            kwargs = {loader.construct_object(key, deep=True): val for key, val in node.value}
            subnodes = sum([val.value if key == '__args' else [val] for key, val in kwargs.items()], [])
            bound = signature.bind(*(kwargs.pop('__args').value if '__args' in kwargs else ()), **kwargs)
        else:
            raise ValueError(f'Invalid node type, {node}')

        # Experimental
        cls.fix_signature(bound)

        # Construct nodes in yaml order
        subnode_values = {n: loader.construct_object(n, deep=True)
                          for n in subnodes}

        for key, val in bound.arguments.items():
            bound.arguments[key] = (
                signature.parameters[key].kind == Parameter.VAR_POSITIONAL
                and (subnode_values[n] for n in val)
                or signature.parameters[key].kind == Parameter.VAR_KEYWORD
                and {name: subnode_values[n] for name, n in val.items()}
                or subnode_values[val]
            )

        # for key, val in bound.arguments.items():
        #     bound.arguments[key] = (
        #         (loader.construct_object(v, deep=True) for v in val)
        #             if signature.parameters[key].kind == Parameter.VAR_POSITIONAL
        #         else {name: loader.construct_object(v, deep=True) for name, v in val.items()}
        #             if signature.parameters[key].kind == Parameter.VAR_KEYWORD
        #         else loader.construct_object(val, deep=True))

        return bound

    @classmethod
    def construct(cls, obj, loader: yaml.Loader, node: yaml.Node, **kw):
        try:
            signature = inspect.signature(obj, follow_wrapped=False)
        except ValueError:
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
        dtype = getattr(torch, suffix)
        if not isinstance(getattr(torch, suffix), torch.dtype):
            raise ValueError(f'In tag "!torch:{suffix}", "{suffix}" is not a valid torch.dtype.')

        return super().construct(torch.tensor, loader, node, dtype=dtype)


class PrefixedStochasticYAMLConstructor(YAMLConstructor):
    @classmethod
    def construct(cls, loader: yaml.Loader, suffix: str, node: yaml.Node, **kwargs):
        return super().construct(stochastic, loader, node, name=suffix)


# sys.version_info >= (3, 8)
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
        # TODO: Needs to be handled better?
        YAMLConstructor.type_to_tag[torch.Tensor] = '!tensor'

        c.add_constructor('!Stochastic', YAMLConstructor.apply(stochastic))
        c.add_multi_constructor('!Stochastic:', PrefixedStochasticYAMLConstructor.construct)

        c.add_constructor('!Sampler', YAMLConstructor.apply(Sampler))

        r: yaml.Resolver = self.resolver
        r.add_path_resolver('!py:Clipppy', [])
        r.add_path_resolver('!py:Guide', ['guide'])
