from __future__ import annotations

import sys
from functools import partial, partialmethod, wraps
from inspect import BoundArguments, cleandoc, Parameter, Signature
from itertools import starmap
from typing import (
    Any, Callable, Iterable, MutableMapping, Optional, Type, TypeVar, Union)
from warnings import warn

from more_itertools import consume, side_effect
from ruamel.yaml import Constructor, MappingNode, Node, ScalarNode, SequenceNode

from . import resolver as resolver_
from .scope import ScopeMixin
from .tagger import NodeTypeMismatchError, TaggerMixin
from ..utils import Sentinel
from ..utils.signatures import get_param_for_name, iter_positional, signature as signature_
from ..utils.typing import Descriptor


_T = TypeVar('_T')


class ClipppyConstructor(ScopeMixin, TaggerMixin, Constructor):
    @wraps(Constructor.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deep_construct = True

    free_signature = Signature((Parameter('args', Parameter.VAR_POSITIONAL),
                                Parameter('kwargs', Parameter.VAR_KEYWORD)))

    resolver: resolver_.ClipppyResolver

    def bind_scalar(self, node: ScalarNode, signature: Signature):
        if node.value is Sentinel.call:
            return signature.bind()

        return (None if (val := self.yaml_constructors[
            self.resolver.resolve(ScalarNode, node.value, (True, False))
        ](self, node)) is None else signature.bind(val))

    def bind_sequence(self, node: SequenceNode, signature: Signature):
        return tuple(map(self.construct_object, starmap(self.tag_from_param, zip(node.value, iter_positional(signature)))))

    def bind_mapping(self, node: MappingNode, signature: Signature):
        pos_params = iter_positional(signature)
        args, kwargs = [], {}
        for key, val in node.value:  # type: Union[Node, str], Union[Node, Iterable]
            if isinstance(key, Node) and key.value == '__args':
                key.tag = self.resolver._mergepos_tag
                warn('Using \'__args\' for parameter expansion is deprecated'
                     ' and will soon be considered an ordinary keyword argument.'
                     f' Consider using \'<\' instead.', FutureWarning)

            if self.resolver.is_pos(key):
                args.append(self.construct_object(self.tag_from_param(val, next(pos_params, None))))
            elif self.resolver.is_mergepos(key):
                if is_default_seq := self.is_default_tagged(val):
                    consume(starmap(self.tag_from_param, zip(val.value, pos_params)))
                val = self.construct_object(val)
                args.extend(val if is_default_seq else side_effect(partial(next, pos_params), val))
            elif self.resolver.is_merge(key):
                if val.tag in (self.resolver.DEFAULT_SCALAR_TAG, self.resolver.DEFAULT_SEQUENCE_TAG):
                    raise NodeTypeMismatchError(f'Expected {MappingNode} for \'{key}\', but got {val}.')
                elif self.is_default_tagged(val):
                    consume(self.tag_from_param(v, get_param_for_name(signature, self.construct_object(k))) for k, v in val.value)
                kwargs.update(self.construct_object(val))
            else:
                key = self.construct_object(key)
                kwargs[key] = self.construct_object(self.tag_from_param(val, get_param_for_name(signature, key)))

        return args, kwargs

    def bind(self, node: Node, signature: Signature = free_signature) -> Optional[BoundArguments]:
        if isinstance(node, ScalarNode):
            return self.bind_scalar(node, signature)
        else:
            if isinstance(node, SequenceNode):
                args = self.bind_sequence(node, signature)
                kwargs = {}
            elif isinstance(node, MappingNode):
                args, kwargs = self.bind_mapping(node, signature)
            else:
                raise TypeError(f'Invalid node type: {node}')

            try:
                return signature.bind(*args, **kwargs)
            except TypeError:
                raise TypeError(Sentinel.sentinel, args, kwargs)

    def construct_object(self, node, deep=True):
        return super().construct_object(node, deep) if isinstance(node, Node) else node

    _type_hook_t = Callable[[Node, 'ClipppyConstructor'], Optional[Union[Node, Any]]]
    type_hooks: MutableMapping[Union[Type, Callable, Any], _type_hook_t] = {}

    @classmethod
    def add_type_hook(cls, obj, hook: _type_hook_t):
        cls.type_hooks[obj] = hook

    @classmethod
    def construct(cls, obj, loader: ClipppyConstructor, node: Node, **kwargs):
        try:
            signature = signature_(obj)
        except (TypeError, ValueError):
            signature = cls.free_signature
        try:
            if obj in loader.type_hooks:
                ret = loader.type_hooks[obj](node, loader)
                if isinstance(ret, Node):
                    node = ret
                elif ret is not None:
                    return ret
            signature = loader.bind(node, signature)
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

            # try:
            return obj(*signature.args, **{**signature.kwargs, **kwargs})
            # except Exception as e:
            #     raise TypeError(f'''Could not instantiate\nobj: {obj}\n*args: {signature.args}\n**kwargs: {signature.kwargs}.''')


    @classmethod
    def construct_prefixed(
            cls, resolver: Callable[[str, MutableMapping[str, Any]], Union[Any, tuple[Any, MutableMapping[str, Any]]]],
            loader: ClipppyConstructor, suffix: str, node: Node, **kwargs):
        obj = resolver(suffix, kwargs)
        if isinstance(obj, tuple) and len(obj) == 2:
            obj, kwargs = obj
        return cls.construct(obj, loader, node, **kwargs)

    @classmethod
    def construct_bound(cls, obj: Descriptor, loader: ClipppyConstructor, node: Node, *args, _cls: Type = None, _func: _constructDescriptorT = construct, **kwargs):
        ldr = next(o for o in (loader, loader.loader) if isinstance(o, _cls or cls))
        return _func.__get__(None, cls)(obj.__get__(ldr, _cls), loader, node, *args, **kwargs)

    @classmethod
    def apply(cls, obj, func: _constructDescriptorT = construct, **kwargs):
        return partial(func.__get__(None, cls), obj, **kwargs)

    apply_prefixed = wraps(apply)(partialmethod(apply, func=construct_prefixed))
    apply_bound = wraps(apply)(partialmethod(apply, func=construct_bound))
    apply_bound_prefixed = wraps(apply)(partialmethod(apply, func=construct_bound, _func=construct_prefixed))


# TODO: python 3.9
_constructT = (
    Union[Callable[[Any, ClipppyConstructor, str, Node], Any],
          Callable[[Any, ClipppyConstructor, Node], Any]]
    if sys.version_info < (3, 9) else
    Union[Callable[[Any, ClipppyConstructor, str, Node, ...], Any],
          Callable[[Any, ClipppyConstructor, Node, ...], Any]]
)
_constructDescriptorT = Descriptor[ClipppyConstructor, _constructT]
