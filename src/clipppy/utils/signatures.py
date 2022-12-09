from __future__ import annotations

import builtins
import inspect
import sys
import types
from functools import wraps, partial

from inspect import Parameter, Signature
from itertools import repeat
from typing import Iterable, Mapping


__all__ = 'is_variadic', 'iter_positional', 'get_kwargs', 'get_param_for_name', 'signature', 'has_var_args'

from ..utils import tryme


def is_variadic(param: Parameter) -> bool:
    return param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)


def has_var_args(s: Signature):
    return any(p.kind is p.VAR_POSITIONAL for p in s.parameters.values())


def iter_positional(s: Signature):
    for param in s.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            yield param
        elif param.kind == param.VAR_POSITIONAL:
            yield from repeat(param)


def get_kwargs(sig: Signature, default=Parameter('_', kind=Parameter.VAR_KEYWORD)) -> Parameter:
    return next(filter(lambda param: param.kind == param.VAR_KEYWORD,
                       reversed(sig.parameters.values())), default)


def get_param_for_name(sig: Signature, name: str):
    return sig.parameters.get(name, get_kwargs(sig))


if sys.version_info >= (3, 10):
    signature = partial(inspect.signature, eval_str=True)
else:
    @wraps(inspect.signature)
    def signature(obj, *args, **kwargs) -> Signature:
        """Poor man's attempt at fixing string annotations..."""
        sig = inspect.signature(obj, *args, **kwargs)
        if isinstance(obj, type):
            obj = obj.__init__
        glob = (obj.__globals__ if isinstance(obj, types.FunctionType)
                else vars(sys.modules[obj.__module__] if hasattr(obj, '__module__') else builtins))
        return sig.replace(
            parameters=[p.replace(annotation=tryme(
                lambda: eval(p.annotation, {'Iterable': Iterable, 'Mapping': Mapping}, glob)
            ))  # wtf, Guido...
                        if isinstance(p.annotation, str) else p
                        for p in sig.parameters.values()],
            return_annotation=eval(sig.return_annotation, {}, glob)
            if isinstance(sig.return_annotation, str) else sig.return_annotation)
