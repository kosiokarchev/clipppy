from __future__ import annotations

import ast
import builtins
import inspect
import re
from itertools import filterfalse
from typing import (Any, Callable, cast, get_args, get_origin, get_type_hints, Iterable, Literal, Mapping, NewType,
                    Optional,
                    Sequence, Tuple, Type,
                    TypedDict,
                    Union)

import click
from more_itertools import always_iterable, always_reversible, first

from .utils import tryme
from .utils.signatures import is_variadic


__all__ = '_File', '_ExistingFile', '_Dir', '_Dir', 'AutoCLI', 'acli'


class EvalParamType(click.ParamType):
    name = 'EVAL'

    def convert(self, value: Union[str, Any], param, ctx) -> Any:
        return (isinstance(value, str) and eval(value, vars(builtins))
                or isinstance(value, ObjectContainer) and value.o
                or value)


_File = NewType('_File', str)
_ExistingFile = NewType('_ExistingFile', str)
_Dir = NewType('_Dir', str)
_ExistingDir = NewType('_ExistingDir', str)

click_types_map = {
    Union: EvalParamType(),
    bool: bool,
    _File: click.Path(file_okay=True, dir_okay=False),
    _ExistingFile: click.Path(exists=True, file_okay=True, dir_okay=False),
    _Dir: click.Path(file_okay=False, dir_okay=True),
    _ExistingDir: click.Path(exists=True, file_okay=False, dir_okay=True),
}


class ParameterWithDoc(inspect.Parameter):
    # Screw them!
    __slots__ = ()
    doc_registry = {}

    @property
    def doc(self) -> str:
        return type(self).doc_registry[id(self)]

    @doc.setter
    def doc(self, value: str):
        type(self).doc_registry[id(self)] = value

    @classmethod
    def from_param(cls, param: inspect.Parameter, doc: str = None) -> 'ParameterWithDoc':
        """Modifies ``param`` in-place!"""
        param.__class__ = cls
        param.doc = doc
        return cast(ParameterWithDoc, param)


def get_attribute_doc(clsasts: Union[Iterable[ast.ClassDef], ast.ClassDef], name) -> Optional[str]:
    return re.sub(r'\n+', '\n', re.sub(r'([^\n])\n([^\n])', r'\1 \2', inspect.cleandoc(first((
        ret for clsast in always_iterable(clsasts, base_type=ast.ClassDef) if clsast is not None
        for members in (clsast.body,)
        for i, member in always_reversible(enumerate(members))
        if ((isinstance(member, ast.AnnAssign) and member.target.id == name
             or isinstance(member, ast.Assign) and name in (node.id for node in ast.walk(member) if isinstance(node, ast.Name)))
            and isinstance(members[i+1], ast.Expr) and isinstance(members[i+1].value, ast.Str))
        for ret in (members[i+1].value.s,)
    ), default='')))) or None


class DocumentedSignature(inspect.Signature):
    parameters: Mapping[str, ParameterWithDoc]

    @classmethod
    def from_callable(cls, obj: Callable[..., Any], *, follow_wrapped: bool = ...) -> 'DocumentedSignature':
        sig = super().from_callable(obj, follow_wrapped=follow_wrapped)
        for param in sig.parameters.values():
            # TODO: Actual doc extraction for function parameters
            ParameterWithDoc.from_param(param, doc=None)

        # noinspection PyTypeChecker
        return sig

    @classmethod
    def from_command(cls, ocls: Union[Type, Any]) -> 'DocumentedSignature':
        if not isinstance(ocls, type):
            ocls = type(ocls)
        annotations = get_type_hints(ocls)

        clsasts = [
            cast(ast.ClassDef, tryme(lambda: ast.parse(inspect.getsource(_cls)).body[0]))
            for _cls in ocls.__mro__
        ]
        return cls([
            ParameterWithDoc.from_param(
                inspect.Parameter(name=name, default=getattr(ocls, name, inspect.Parameter.empty),
                                  annotation=ann, kind=inspect.Parameter.KEYWORD_ONLY),
                doc=get_attribute_doc(clsasts, name=name))
            for name, ann in annotations.items()
        ])


class ObjectContainer:
    def __init__(self, o):
        self.o = o

    def __repr__(self):
        return repr(self.o)


class AutoCLI:
    def __init__(self, **kwargs):
        self.option_kwargs = {
            param.name: kwargs.get(param.name, param.default)
            for param in inspect.signature(click.Option).parameters.values()
            if not is_variadic(param)
        }
        self.argument_kwargs = {
            param.name: kwargs.get(param.name, param.default)
            for param in inspect.signature(click.Argument).parameters.values()
            if not is_variadic(param)
        }

        self._registry = {}

    _atc_type = TypedDict('_atc_type', dict(type=Union[click.ParamType, Tuple], multiple=bool), total=False)

    def annotation_to_click(self, ann) -> _atc_type:
        try:
            origin = get_origin(ann) or ann
            if origin in click_types_map:
                return {'type': click_types_map[origin]}

            elif origin is Literal:
                choices = [None if a is None else str(a) for a in get_args(ann)]
                return {'type': click.Choice(choices),
                        'metavar': f'[{"|".join(filterfalse(lambda x: (x is None), choices))}]'}
            elif origin is Tuple:
                return {'type': tuple(self.annotation_to_click(a)['type'] for a in get_args(ann))}
            elif issubclass(origin, (get_origin(Iterable), get_origin(Sequence)))\
                    and not issubclass(origin, str)\
                    and not issubclass(origin, Mapping):
                return {'type': self.annotation_to_click(get_args(ann)[0])['type'], 'multiple': True}
            else:
                return {'type': click.types.convert_type(origin, EvalParamType())}
        except Exception as e:
            print(e, ann)
            return {}

    def parameter_to_click(self, param: ParameterWithDoc) -> Optional[click.Parameter]:
        if param.annotation is click.Context or is_variadic(param):
            return None

        name = param.name.replace('_', '-')
        is_bool = param.annotation is bool or isinstance(param.default, bool)
        annkwargs = self.annotation_to_click(param.annotation)
        return (
            param.default is not param.empty
            and click.Option(**{
                **self.option_kwargs, **annkwargs, **dict(
                    param_decls=[(is_bool and param.default) and f'--{name}/--no-{name}' or f'--{name}', param.name],
                    is_flag=is_bool,
                    default=param.default if not callable(param.default) else ObjectContainer(param.default),
                    help=(param.doc or self.option_kwargs.get('help', '') or '').split('\n', 1)[0]
                )})
            or click.Argument(**{**self.argument_kwargs, **annkwargs, 'required': True, 'param_decls': [name]})
        )

    def sigs_to_click(self, *sigs: DocumentedSignature):
        return {
            param.name: click_param
            for sig in sigs
            for param in sig.parameters.values()
            for click_param in (self.parameter_to_click(param),)
            if click_param is not None
        }

    def annotated_class(self, obj, follow_wrapped=True) -> Mapping[str, click.Parameter]:
        return self.sigs_to_click(
            DocumentedSignature.from_command(obj),
            DocumentedSignature.from_callable(obj, follow_wrapped=follow_wrapped)
        )

    def function(self, func, follow_wrapped=True) -> Mapping[str, click.Parameter]:
        return self.sigs_to_click(
            DocumentedSignature.from_callable(func, follow_wrapped=follow_wrapped)
        )


acli = AutoCLI()
