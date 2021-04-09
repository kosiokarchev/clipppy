import inspect
from _warnings import warn
from functools import partial
from inspect import BoundArguments, Parameter, Signature
from itertools import repeat, starmap
from typing import Callable, Dict, get_origin, Iterable, Mapping, Optional, Type, TypeVar, Union

from more_itertools import consume, side_effect
from ruamel import yaml as yaml

from ..utils import valueiter
from ..utils.signatures import get_param_for_name


_T = TypeVar('_T')


class NodeTypeMismatchError(Exception):
    pass


class YAMLConstructor:
    free_signature = Signature((Parameter('args', Parameter.VAR_POSITIONAL),
                                Parameter('kwargs', Parameter.VAR_KEYWORD)))

    type_to_tag: Dict[Type, str] = {}

    strict_node_type = True

    @classmethod
    def infer_single_tag(cls, node: Union[yaml.Node, _T], param: Parameter) -> Union[yaml.Node, _T]:
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
                        else get_origin(param.annotation) or None)

            if not inspect.isclass(hint):
                # TODO: Maybe handle Union??
                # noinspection PyTypeChecker
                return

            hint_is_builtin = any(hint.__module__.startswith(mod)
                                  for mod in ('builtins', 'typing', 'collections'))
            hint_is_str = issubclass(hint, str)
            hint_is_callable = issubclass(hint, type) or (not isinstance(hint, type) and issubclass(hint, Callable))
            target_nodetype = (
                    (hint_is_callable or hint_is_str) and yaml.ScalarNode
                    or hint_is_builtin and issubclass(hint, Mapping) and yaml.MappingNode
                    or hint_is_builtin and issubclass(hint, Iterable) and yaml.SequenceNode
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
                        signature: Signature = free_signature) -> Optional[BoundArguments]:
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
            for key, val in node.value:  # type: Union[yaml.Node, str], Union[yaml.Node, Iterable]
                if key.tag == 'tag:yaml.org,2002:merge':
                    key.tag = loader.DEFAULT_SCALAR_TAG
                key = construct_object(key)
                if key == '__args':
                    key = cls._args_key
                    warn(f'Using \'__args\' for parameter expansion is deprecated and will soon be considered an ordinary keyword argument.'
                         f' Consider using \'{cls._args_key}\' instead.', FutureWarning)
                if key == cls._args_key:
                    is_default_seq = val.tag == loader.DEFAULT_SEQUENCE_TAG
                    if is_default_seq:
                        consume(starmap(cls.infer_single_tag, zip(val.value, pos_params)))
                    val = construct_object(val)
                    args.extend(val if is_default_seq else side_effect(partial(next, pos_params), val))
                elif key == cls._kwargs_key:
                    if val.tag in (loader.DEFAULT_SCALAR_TAG, loader.DEFAULT_SEQUENCE_TAG):
                        raise NodeTypeMismatchError(f'Expected {yaml.MappingNode} for \'{key}\', but got {val}.')
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
