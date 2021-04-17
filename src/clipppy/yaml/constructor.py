import collections.abc
from _warnings import warn
from functools import partial
from inspect import BoundArguments, cleandoc, isclass, Parameter, Signature
from itertools import chain, repeat, starmap
from types import GenericAlias
from typing import (
    Any, Callable, Dict, get_args, get_origin, Iterable, Mapping, Optional, Type, TypeVar, Union)

from more_itertools import consume, side_effect
from ruamel import yaml as yaml

from ..utils import Sentinel
from ..utils.signatures import get_param_for_name, iter_positional, signature as signature_


_T = TypeVar('_T')
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


class PerKeyDefaultDict(dict[_KT, _VT]):
    default_factory: Optional[Callable[[_KT], _VT]] = None

    def __init__(self, default_factory: Callable[[_KT], _VT] = None, *args, **kwargs):
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __missing__(self, key: _KT) -> _VT:
        if self.default_factory is None:
            raise KeyError(key)
        res = self.default_factory(key)
        self[key] = res
        return res


class NodeTypeMismatchError(Exception):
    pass


class YAMLConstructor:
    free_signature = Signature((Parameter('args', Parameter.VAR_POSITIONAL),
                                Parameter('kwargs', Parameter.VAR_KEYWORD)))

    _pos_tag = 'tag:yaml.org,2002:pos'
    _mergepos_tag = 'tag:yaml.org,2002:mergepos'
    _merge_tag = 'tag:yaml.org,2002:merge'

    _default_tags = {
        yaml.ScalarNode: yaml.Resolver.DEFAULT_SCALAR_TAG,
        yaml.SequenceNode: yaml.Resolver.DEFAULT_SEQUENCE_TAG,
        yaml.MappingNode: yaml.Resolver.DEFAULT_MAPPING_TAG,
    }

    @classmethod
    def default_tag_for_node(cls, node: yaml.Node):
        return next((val for key, val in cls._default_tags.items() if isinstance(node, key)), None)

    @classmethod
    def default_tags_not_for_node(cls, node: yaml.Node):
        return {val for key, val in cls._default_tags.items() if not isinstance(node, key)}

    type_to_tag: Dict[Type, str] = PerKeyDefaultDict(lambda key: f'!py:{key.__module__}.{key.__name__}')

    strict_node_type = True

    @classmethod
    def tag_node(cls, node: yaml.Node, hint):
        if isinstance(node.tag, str) and not node.tag.startswith('tag:yaml.org'):
            return node

        hint_is_builtin = hint.__module__.startswith(('builtins', 'typing', 'collections'))
        hint_is_str = issubclass(hint, str)
        hint_is_callable = hint is collections.abc.Callable or issubclass(hint, type)
        target_nodetype = (
            (hint_is_callable or hint_is_str) and yaml.ScalarNode
            or hint_is_builtin and issubclass(hint, Mapping) and yaml.MappingNode
            or hint_is_builtin and issubclass(hint, Iterable) and yaml.SequenceNode
            or yaml.Node
        )

        if not isinstance(node, target_nodetype):
            exc = NodeTypeMismatchError(f'Expected {target_nodetype} for {hint}, but got {node}.')
            if cls.strict_node_type:
                raise exc
            else:
                warn(str(exc), RuntimeWarning)
        if not hint_is_str and (not hint_is_builtin or target_nodetype in (yaml.ScalarNode, yaml.Node)):
            if hint_is_callable:
                node.tag = f'!py:{node.value}'
                node.value = ''
            else:
                node.tag = cls.type_to_tag[hint]
                if node.anchor and not node.value:
                    node.value = Sentinel.call

        return node

    @classmethod
    def tag_from_hint(cls, node: yaml.Node, hint):
        origin = hint if isclass(hint) else get_origin(hint)

        if isclass(origin):
            if isinstance(origin, GenericAlias):
                origin = get_origin(origin)

            consume(starmap(cls.tag_from_hint, (
                chain(*starmap(zip, zip(node.value, repeat(get_args(hint)))))
                if issubclass(origin, collections.abc.Mapping) and isinstance(node, yaml.MappingNode) else
                zip(node.value, repeat(args[0]) if len(args := get_args(hint)) == 1 else args)
                if issubclass(origin, collections.abc.Iterable) and isinstance(node, yaml.SequenceNode) else
                ()
            )))

            return cls.tag_node(node, origin)
        else:  # Union, etc...
            # TODO: Maybe handle Union??
            return node

    @classmethod
    def tag_from_param(cls, node: yaml.Node, param: Optional[Parameter]) -> yaml.Node:
        if param is None or isinstance(node.tag, str) and not node.tag.startswith('tag:yaml.org'):
            return node

        if param.annotation is param.empty:
            if param.default in (param.empty, None):
                return node
            else:
                hint = type(param.default)
        else:
            hint = param.annotation

        return cls.tag_from_hint(node, hint)

    @classmethod
    def bind_scalar(cls, node: yaml.ScalarNode, loader: yaml.Loader, signature: Signature):
        if node.value is Sentinel.call:
            return signature.bind()

        # try:
        #     # This sometimes fails because of ruamel/yaml/resolver.py line 370
        #     node.tag = loader.resolver.resolve(yaml.ScalarNode, node.value, [True, False])
        # except IndexError:
        #     node.tag = cls.default_tag_for_node(node)

        node.tag = loader.resolver.resolve(yaml.ScalarNode, node.value, [True, False])
        val = loader.yaml_constructors[node.tag](loader, node)
        return None if val is None else signature.bind(val)

    @classmethod
    def bind_sequence(cls, node: yaml.SequenceNode, constructor: Callable[[yaml.Node], Any], signature: Signature):
        return tuple(map(constructor, starmap(cls.tag_from_param, zip(node.value, iter_positional(signature))))), {}

    @classmethod
    def bind_mapping(cls, node: yaml.MappingNode, constructor: Callable[[yaml.Node], Any], signature: Signature):
        pos_params = iter_positional(signature)
        args, kwargs = [], {}
        for key, val in node.value:  # type: Union[yaml.Node, str], Union[yaml.Node, Iterable]
            if key.value == '__args':
                key.tag = cls._mergepos_tag
                warn('Using \'__args\' for parameter expansion is deprecated'
                     ' and will soon be considered an ordinary keyword argument.'
                     f' Consider using \'<\' instead.', FutureWarning)

            if key.tag == cls._pos_tag:
                args.append(constructor(cls.tag_from_param(val, next(pos_params, None))))
            elif key.tag == cls._mergepos_tag:
                is_default_seq = val.tag == cls.default_tag_for_node(node)
                if is_default_seq:
                    consume(starmap(cls.tag_from_param, zip(val.value, pos_params)))
                val = constructor(val)
                args.extend(val if is_default_seq else side_effect(partial(next, pos_params), val))
            elif key.tag == cls._merge_tag:
                if val.tag in cls.default_tags_not_for_node(node):
                    raise NodeTypeMismatchError(f'Expected {yaml.MappingNode} for \'{key}\', but got {val}.')
                elif val.tag == cls.default_tag_for_node(node):
                    consume(cls.tag_from_param(v, get_param_for_name(signature, constructor(k))) for k, v in val.value)
                kwargs.update(constructor(val))
            else:
                key = constructor(key)
                kwargs[key] = constructor(cls.tag_from_param(val, get_param_for_name(signature, key)))

        return args, kwargs


    @classmethod
    def bind(cls, node: yaml.Node, loader: yaml.Loader, signature: Signature = free_signature) -> Optional[BoundArguments]:
        if isinstance(node, yaml.ScalarNode):
            return cls.bind_scalar(node, loader, signature)
        else:
            construct_object = partial(loader.construct_object, deep=True)
            if isinstance(node, yaml.SequenceNode):
                args, kwargs = cls.bind_sequence(node, construct_object, signature)
            elif isinstance(node, yaml.MappingNode):
                args, kwargs = cls.bind_mapping(node, construct_object, signature)
            else:
                raise TypeError(f'Invalid node type: {node}')

            try:
                return signature.bind(*args, **kwargs)
            except TypeError:
                raise TypeError(Sentinel.sentinel, args, kwargs)

    @classmethod
    def construct(cls, obj, loader: yaml.Loader, node: yaml.Node, **kw):
        try:
            signature = signature_(obj)
        except (TypeError, ValueError):
            signature = cls.free_signature

        try:
            signature = cls.bind(node, loader, signature)
        except TypeError as e:
            if e.args and e.args[0] is Sentinel.sentinel:
                raise TypeError(cleandoc(f'''
                    Cannot bind:
                      *args: {e.args[1]}
                      *kwargs: {e.args[2]}
                    to {obj}: {signature!s}''')) from None
            else:
                raise
        if signature is None:
            return obj
        else:
            signature.apply_defaults()
            signature.arguments.update(kw)

            try:
                return obj(*signature.args, **signature.kwargs)
            except:
                raise TypeError(f'''Could not instantiate\nobj: {obj}\n*args: {signature.args}\n**kwargs: {signature.kwargs}.''')

    @classmethod
    def apply(cls, obj):
        return partial(cls.construct, obj)
