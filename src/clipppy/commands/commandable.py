from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any, get_type_hints, Mapping, Optional, Type

from . import command


class ProxyDict(dict):
    def __init__(self, obj, keys):
        super().__init__()
        self._keys = keys
        self.obj = obj

    def keys(self):
        return self._keys

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def __getitem__(self, item):
        return getattr(self.obj, item)


class Commandable:
    @property
    @lru_cache()
    def commands(self) -> Mapping[str, Any]:
        return get_type_hints(type(self))

    def get_cmd_cls(self, name: str) -> Optional[Type[command.Command]]:
        if (cmd := type(self).__annotations__.get(name, None)) is not None:
            if isinstance(cmd, str):
                cmd = eval(cmd, __globals=sys.modules[self.__module__].__dict__, __locals=dict(vars(type(self))))
            if issubclass(cmd, command.Command):
                return cmd
        return None

    def register_cmd_cls(self, name: str, cls: Type[command.Command]):
        type(self).__annotations__[name] = cls

    def __setattr__(self, key, value):
        if isinstance(value, command.Command):
            value.commander = self
            value.boundkwargs = ProxyDict(self, ('model', 'guide'))
        elif isinstance(value, dict) and (cmd_cls := self.get_cmd_cls(key)) is not None:
            return self.__setattr__(key, cmd_cls(**value))

        return super().__setattr__(key, value)

    def __getattr__(self, name: str):
        if (cmd := self.get_cmd_cls(name)) is not None:
            setattr(self, name, cmd())
            return getattr(self, name)

        raise AttributeError(name)
