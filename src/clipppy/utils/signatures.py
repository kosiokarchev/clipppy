__all__ = 'is_variadic', 'get_param_for_name'

import inspect
import sys
import types
from functools import wraps

from inspect import Parameter, Signature
from itertools import repeat


def is_variadic(param: Parameter) -> bool:
    return param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)


def iter_positional(s: Signature):
    for param in s.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            yield param
        elif param.kind == param.VAR_POSITIONAL:
            yield from repeat(param)


def get_kwargs(signature: Signature, default=Parameter('_', kind=Parameter.VAR_KEYWORD)) -> Parameter:
    return next(filter(lambda param: param.kind == param.VAR_KEYWORD,
                       reversed(signature.parameters.values())), default)


def get_param_for_name(signature: Signature, name: str):
    return signature.parameters.get(name, get_kwargs(signature))


@wraps(inspect.signature)
def signature(obj, *args, **kwargs) -> Signature:
    """Poor man's attempt at fixing string annotations..."""
    sig = inspect.signature(obj, *args, **kwargs)
    glob = (obj.__globals__ if isinstance(obj, types.FunctionType)
            else vars(sys.modules[obj.__module__]))
    return sig.replace(
        parameters=[p.replace(annotation=eval(p.annotation, {}, glob))
                    if isinstance(p.annotation, str) else p
                    for p in sig.parameters.values()],
        return_annotation=eval(sig.return_annotation, {}, glob)
        if isinstance(sig.return_annotation, str) else sig.return_annotation)
