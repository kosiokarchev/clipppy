__all__ = 'is_variadic', 'get_param_for_name'

from inspect import Parameter, Signature


def is_variadic(param: Parameter) -> bool:
    return param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)


def get_kwargs(signature: Signature, default=Parameter('_', kind=Parameter.VAR_KEYWORD)) -> Parameter:
    return next(filter(lambda param: param.kind == param.VAR_KEYWORD,
                       reversed(signature.parameters.values())), default)


def get_param_for_name(signature: Signature, name: str):
    return signature.parameters.get(name, get_kwargs(signature))
