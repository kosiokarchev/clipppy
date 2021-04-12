from dataclasses import dataclass
from importlib import import_module
from operator import attrgetter

from more_itertools import rlocate
from ruamel.yaml import Loader, Node

from .constructor import YAMLConstructor
from .. import yaml


__all__ = 'PyYAMLConstructor',


@dataclass
class PyYAMLConstructor(YAMLConstructor):
    _yaml: 'yaml.ClipppyYAML'

    @property
    def scope(self):
        return self._yaml.scope

    def resolve(self, name: str):
        try:
            # TypeError: "globals must be a real dict; try eval(expr, {}, mapping)" :)
            return eval(name, {}, self.scope)
        except (NameError, AttributeError) as err:
            for i in rlocate(name, '.'.__eq__):
                try:
                    return attrgetter(name[i+1:])(import_module(name[:i]))
                except ModuleNotFoundError as err:
                    pass
            raise err from None

    def construct(self, loader: Loader, suffix: str, node: Node, **kwargs):
        return super().construct(self.resolve(suffix), loader, node, **kwargs)
