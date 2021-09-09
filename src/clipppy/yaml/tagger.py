from __future__ import annotations

import collections.abc
from inspect import isclass, Parameter
from itertools import chain, repeat, starmap
from types import GenericAlias
from typing import (
    Callable, ClassVar, Dict, get_args, get_origin, Iterable, Mapping, MutableMapping, Optional, Type, TypeVar, Union)
from warnings import warn

from more_itertools import consume
from ruamel.yaml import MappingNode, Node, ScalarNode, SequenceNode

from . import resolver
from ..utils import Sentinel


_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


# TODO: python 3.9: Dict -> dict
class PerKeyDefaultDict(Dict[_KT, _VT]):
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


class TaggerMixin:
    type_to_tag: ClassVar[MutableMapping[Type, str]] = PerKeyDefaultDict(lambda key: f'!py:{key.__module__}.{key.__name__}')

    resolver: resolver.ClipppyResolver

    def is_default_tagged(self, node: Node):
        return (isinstance(node.tag, str) and not isinstance(node.value, Sentinel)
                and node.tag == self.resolver.resolve(type(node), node.value, (True, False)))

    def is_explicit_tagged(self, node: Node):
        return not self.is_default_tagged(node)


    strict_node_type = True

    def tag_node(self, node: Node, hint):
        if self.is_explicit_tagged(node):
            return node

        hint_is_builtin = hint.__module__.startswith(('builtins', 'typing', 'collections'))
        hint_is_str = issubclass(hint, str)
        hint_is_callable = hint is collections.abc.Callable or issubclass(hint, type)
        target_nodetype = (
                (hint_is_callable or hint_is_str) and ScalarNode
                or hint_is_builtin and issubclass(hint, Mapping) and MappingNode
                or hint_is_builtin and issubclass(hint, Iterable) and SequenceNode
                or Node
        )

        if not isinstance(node, target_nodetype):
            exc = NodeTypeMismatchError(f'Expected {target_nodetype} for {hint}, but got {node}.')
            if self.strict_node_type:
                raise exc
            else:
                warn(str(exc), RuntimeWarning)
        if not hint_is_str and (not hint_is_builtin or target_nodetype in (ScalarNode, Node)):
            if hint_is_callable:
                node.tag = f'!py:{node.value}'
                node.value = ''
            else:
                node.tag = self.type_to_tag[hint]
                if node.anchor and not node.value:
                    node.value = Sentinel.call

        return node

    def tag_from_hint(self, node: Node, hint):
        origin = hint if isclass(hint) else get_origin(hint)

        if isclass(origin):
            if isinstance(origin, GenericAlias):
                origin = get_origin(origin)

            consume(starmap(self.tag_from_hint, (
                chain(*starmap(zip, zip(node.value, repeat(get_args(hint)))))
                if issubclass(origin, collections.abc.Mapping) and isinstance(node, MappingNode) else
                zip(node.value, repeat(args[0]) if len(args := get_args(hint)) == 1 else args)
                if issubclass(origin, collections.abc.Iterable) and isinstance(node, SequenceNode) else
                ()
            )))
        elif origin is Union:
            if len(args := get_args(hint)) == 2 and args[1] is type(None):
                origin = args[0]  # Optional[...]
            else:  # TODO: Maybe handle Union??
                return node
        else:
            return node

        return self.tag_node(node, origin)

    def tag_from_param(self, node: Node, param: Optional[Parameter]):
        if not isinstance(node, Node) or param is None or self.is_explicit_tagged(node):
            return node

        if param.annotation is param.empty:
            if param.default in (param.empty, None):
                return node
            else:
                hint = type(param.default)
        else:
            hint = param.annotation

        return self.tag_from_hint(node, hint)
