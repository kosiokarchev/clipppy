import ast
import dataclasses
from importlib import import_module
from typing import Dict, Union

from ruamel.yaml import Loader, Node

from .constructor import YAMLConstructor
from .. import yaml


__all__ = 'PyYAMLConstructor',


@dataclasses.dataclass
class PyYAMLConstructor(YAMLConstructor):
    _yaml: 'yaml.ClipppyYAML'

    @property
    def scope(self):
        return self._yaml.scope

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
                        if isinstance(stmt, ast.Import):
                            for names in stmt.names:
                                if names.asname is None:
                                    res[names.name] = __import__(names.name)
                                else:
                                    res[names.asname] = import_module(names.name)
                        elif isinstance(stmt, ast.ImportFrom):
                            mod = import_module(stmt.module)
                            for names in stmt.names:
                                if names.name == '*':
                                    for name in getattr(mod, '__all__',
                                                        [key for key in vars(mod) if not key.startswith('_')]):
                                        res[name] = getattr(mod, name)
                                else:
                                    res[names.asname if names.asname is not None else names.name] = getattr(mod, names.name)
                except SyntaxError as e:
                    if not spec.startswith('import'):
                        spec = 'import ' + spec
                        continue  # retry
                    else:
                        raise e
                else:
                    break
        self.scope.update(res)

    def resolve(self, name: str):
        if '.' in name:
            mods = name.split('.')
            name = mods.pop()

            ret = self.scope.get(mods[0], None)
            if ret is not None:
                for r in mods[1:]:
                    ret = getattr(ret, r)
            else:
                ret = import_module('.'.join(mods))

            ret = getattr(ret, name)
        else:
            ret = self.scope.get(name, None)
            if ret is None:
                raise NameError(f'name \'{name}\' is not defined')

        return ret

    def construct(self, loader: Loader, suffix: str, node: Node, **kwargs):
        return super().construct(self.resolve(suffix), loader, node, **kwargs)
