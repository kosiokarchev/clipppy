from __future__ import annotations

from functools import lru_cache
from typing import Any, get_type_hints, Mapping, Optional, Type

from .command import Command


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

    def get_cmd_cls(self, name: str) -> Optional[Type[Command]]:
        # sys.version_info >= (3, 8)
        cmd = self.commands.get(name, None)
        return cmd if cmd is not None and issubclass(cmd, Command) else None

    def register_cmd_cls(self, name: str, cls: Type[Command]):
        type(self).__annotations__[name] = cls

    def __setattr__(self, key, value):
        if isinstance(value, Command):
            value.boundkwargs = ProxyDict(self, ('model', 'guide'))
            # value.boundkwargs = dict(model=self.model, guide=self.guide)
        elif isinstance(value, dict):
            # sys.version_info >= (3, 8)
            cmd_cls = self.get_cmd_cls(key)
            if cmd_cls is not None:
                return self.__setattr__(key, cmd_cls(**value))

        super().__setattr__(key, value)

    def __getattr__(self, name: str):
        # sys.version_indo > (3, 8)
        cmd = self.get_cmd_cls(name)
        if cmd is not None:
            setattr(self, name, cmd())
            return getattr(self, name)

        raise AttributeError(name)
