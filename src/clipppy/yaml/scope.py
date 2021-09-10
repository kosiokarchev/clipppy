from __future__ import annotations

import builtins as __builtins__
import ast
import threading
from collections import ChainMap
from importlib import import_module
from operator import attrgetter
from typing import Any, ClassVar, Dict, Mapping, MutableMapping, Protocol, Union

from more_itertools import rlocate


__all__ = 'ScopeMixin',

__builtins__ = vars(__builtins__)  # ensure this, since the standard does not guarantee it


# noinspection PyUnresolvedReferences
def _make_globals():
    import torch
    import numpy as np

    return locals()


class ScopeMixin:
    builtins: ClassVar[Mapping[str, Any]] = ChainMap(_make_globals(), __builtins__)

    class TLocal(Protocol):
        """"""
        scope: MutableMapping[str, Any]

    _tlocal: TLocal = threading.local()

    @property
    def scope(self):
        if not hasattr(self._tlocal, 'scope'):
            self.scope = {}
        return self._tlocal.scope

    @scope.setter
    def scope(self, scope: Mapping[str, Any]):
        self._tlocal.scope = ChainMap(scope, self.builtins)

    # sys.version_info >= (3, 8) TODO: true?
    # spec: Union[str, TypedDict('', **{'from': str, 'import': Union[str, Sequence[str]]}, total=False)]
    def import_(self, *specs: Union[str, Dict]):
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


    def resolve_name(self, name: str, kwargs):
        try:
            return eval(name, {}, self.scope)
        except (NameError, AttributeError) as err:
            for i in rlocate(name, '.'.__eq__):
                try:
                    return attrgetter(name[i + 1:])(import_module(name[:i]))
                except ModuleNotFoundError as err:
                    pass
            raise err from None
