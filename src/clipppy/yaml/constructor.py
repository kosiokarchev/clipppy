import inspect
from _warnings import warn
from functools import partial
from inspect import BoundArguments, Parameter, Signature
from itertools import repeat, starmap
from typing import Callable, Dict, get_origin, Iterable, Mapping, Optional, Type, TypeVar, Union

from more_itertools import consume, side_effect
from ruamel import yaml as yaml

from ..utils.signatures import get_param_for_name


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
    _sentinel = object()
    free_signature = Signature((Parameter('args', Parameter.VAR_POSITIONAL),
                                Parameter('kwargs', Parameter.VAR_KEYWORD)))

    type_to_tag: Dict[Type, str] = PerKeyDefaultDict(lambda key: f'!py:{key.__module__}.{key.__name__}')

    strict_node_type = True

    @classmethod
    def infer_single_tag(cls, node: yaml.Node, param: Parameter) -> yaml.Node:
        try:
            if param.annotation is param.empty:
                if param.default in (param.empty, None):
                    # noinspection PyTypeChecker
                    return
                else:
                    hint = type(param.default)
            else:
                hint = (param.annotation if inspect.isclass(param.annotation)
                        else get_origin(param.annotation) or None)

            if not inspect.isclass(hint):
                # TODO: Maybe handle Union??
                # noinspection PyTypeChecker
                return

            hint_is_builtin = any(hint.__module__.startswith(mod)
                                  for mod in ('builtins', 'typing', 'collections'))
            hint_is_str = issubclass(hint, str)
            hint_is_callable = (hint in (Callable, get_origin(Callable)) or issubclass(hint, type))
            target_nodetype = (
                (hint_is_callable or hint_is_str) and yaml.ScalarNode
                or hint_is_builtin and issubclass(hint, Mapping) and yaml.MappingNode
                or hint_is_builtin and issubclass(hint, Iterable) and yaml.SequenceNode
                or yaml.Node
            )

            if not isinstance(node, target_nodetype):
                raise NodeTypeMismatchError(f'Expected {target_nodetype} for {param}, but got {node}.')
            if (not hint_is_str
                    and (not hint_is_builtin or target_nodetype in (yaml.ScalarNode, yaml.Node))
                    and node.tag.startswith('tag:yaml.org')):
                if hint_is_callable:
                    node.tag = f'!py:{node.value}'
                    node.value = ''
                else:
                    node.tag = cls.type_to_tag[hint]
                    if node.anchor and not node.value:
                        node.value = cls._sentinel
        except Exception as e:
            if cls.strict_node_type and isinstance(e, NodeTypeMismatchError):
                raise
            warn(str(e), RuntimeWarning)
        finally:
            return node

    _args_tag = 'tag:yaml.org,2002:mergepos'
    _kwargs_tag = 'tag:yaml.org,2002:merge'

    @classmethod
    def get_args_kwargs(cls, loader: yaml.Loader, node: yaml.Node,
                        signature: Signature = free_signature) -> Optional[BoundArguments]:
        construct_object = partial(loader.construct_object, deep=True)
        if isinstance(node, yaml.ScalarNode):
            if node.value is cls._sentinel:
                return signature.bind()
            try:
                # This sometimes fails because of ruamel/yaml/resolver.py line 370
                node.tag = loader.resolver.resolve(yaml.ScalarNode, node.value, [True, False])
            except IndexError:
                node.tag = loader.DEFAULT_SCALAR_TAG
            val = loader.yaml_constructors[node.tag](loader, node)
            return None if val is None else signature.bind(val)
        else:
            pos_params = (p for param in signature.parameters.values()
                          for p in ((param,) if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
                                    else repeat(param) if param.kind == param.VAR_POSITIONAL else ()))
            if isinstance(node, yaml.SequenceNode):
                return signature.bind(*map(construct_object, starmap(cls.infer_single_tag, zip(node.value, pos_params))))
            elif isinstance(node, yaml.MappingNode):
                args, kwargs = [],  {}
                for key, val in node.value:  # type: Union[yaml.Node, str], Union[yaml.Node, Iterable]
                    if key.value == '__args':
                        key.tag = cls._args_tag
                        warn('Using \'__args\' for parameter expansion is deprecated'
                             ' and will soon be considered an ordinary keyword argument.'
                             f' Consider using \'<\' instead.', FutureWarning)
                    if key.tag == cls._args_tag:
                        is_default_seq = val.tag == loader.DEFAULT_SEQUENCE_TAG
                        if is_default_seq:
                            consume(starmap(cls.infer_single_tag, zip(val.value, pos_params)))
                        val = construct_object(val)
                        args.extend(val if is_default_seq else side_effect(partial(next, pos_params), val))
                    elif key.tag == cls._kwargs_tag:
                        if val.tag in (loader.DEFAULT_SCALAR_TAG, loader.DEFAULT_SEQUENCE_TAG):
                            raise NodeTypeMismatchError(f'Expected {yaml.MappingNode} for \'{key}\', but got {val}.')
                        elif val.tag == loader.DEFAULT_MAPPING_TAG:
                            consume(cls.infer_single_tag(v, get_param_for_name(signature, construct_object(k))) for k, v in val.value)
                        kwargs.update(construct_object(val))
                    else:
                        key = construct_object(key)
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
