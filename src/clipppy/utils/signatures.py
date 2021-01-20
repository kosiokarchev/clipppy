import inspect

__all__ = ('is_variadic',)


def is_variadic(param: inspect.Parameter) -> bool:
    return param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
