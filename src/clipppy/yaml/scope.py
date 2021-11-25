from __future__ import annotations

import builtins as __builtins__
import ast
import threading
from collections import ChainMap
from importlib import import_module
from operator import attrgetter
from typing import Any, ClassVar, Iterable, Mapping, MutableMapping, Protocol, TypedDict, Union

from more_itertools import rlocate


__all__ = 'ScopeMixin',


_import_DictT = TypedDict('_import_DictT', {'from': str, 'import': Union[str, Iterable[str]]}, total=False)


class ScopeMixin:
    builtins: ClassVar[Mapping[str, Any]] = ChainMap(vars(__builtins__))

    _scope: MutableMapping[str, Any] = None

    @property
    def scope(self):
        if self._scope is None:
            self.scope = {}
        return self._scope

    @scope.setter
    def scope(self, value: Mapping[str, Any]):
        self._scope = ChainMap(value, self.builtins)

    def import_(self, *specs: Union[str, _import_DictT]):
        res = {}
        for spec in specs:
            if not isinstance(spec, str):
                spec = f'from {spec["from"]} import {spec["import"] if isinstance(spec["import"], str) else ", ".join(spec["import"])}'
            while True:
                try:
                    for stmt in ast.parse(spec).body:
                        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                            exec(compile(ast.Module([stmt], []), '', 'exec'), {}, res)
                        else:
                            raise SyntaxError('Only import (from) statements are allowed.')
                    break
                except SyntaxError as e:
                    if not spec.startswith('import'):
                        spec = 'import ' + spec
                        continue  # retry
                    else:
                        raise e
        self.scope.update(res)

    # noinspection PyUnusedLocal
    def resolve_name(self, name: str, kwargs):
        try:
            return eval(name, {}, self.scope)
        except (NameError, AttributeError) as err:
            for i in rlocate(name, '.'.__eq__):
                try:
                    return attrgetter(name[i + 1:])(import_module(name[:i]))
                except ModuleNotFoundError as err1:
                    err = err1
            raise err from None
