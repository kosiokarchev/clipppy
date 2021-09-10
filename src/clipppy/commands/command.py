from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from contextlib import nullcontext
from functools import lru_cache
from typing import Any, ContextManager, get_type_hints, Iterable, Union

import pyro

from ..utils import Sentinel


class Command(ABC):
    """
    An abstract base class for commands.

    Defines an interface that allows manipulating parameters of the command in
    three distinct ways. Firstly, the *class* object itself provides the
    defaults. Next, when an instance is created, the constructor takes
    arbitrary keyword arguments that will be set as *instance* attributes
    overriding the defaults. Finally, when a class instance is called, any
    possible parameters that are included in the :python:``**kwargs`` parameter
    are extracted and set on the instance before execution of the
    `~Command.forward` method, which should be overridden to provide the actual
    command implementation.

    Commands also support "binding" keyword parameters in much the same way
    as `functools.partial`, through the `boundkwargs` property. The provided
    keywords are then always included in the `~Command.forward` call, if the
    forward call has explicit parameters with the same names.
    """

    boundkwargs: dict
    """A dictionary of values to be forwarded to each call of `forward`,
       if there are exact name matches in its signature."""

    no_call = Sentinel.no_call
    """Special value to be used to indicate that instead of calling an object,
       it should be returned as is."""

    @property
    @lru_cache()
    def attr_names(self) -> list[str]:
        return list(get_type_hints(type(self)).keys())

    def setattr(self, kwargs: dict[str, Any]):
        for key in list(key for key in kwargs if key in self.attr_names):
            setattr(self, key, kwargs.pop(key))
        return kwargs

    def __init__(self, **kwargs):
        self.setattr(kwargs)
        self.boundkwargs: dict = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        oldkwargs = {name: getattr(self, name) for name in self.attr_names}
        kwargs = self.setattr(kwargs)
        allowed = inspect.signature(self.forward).parameters
        try:
            return self.forward(*args, **{
                **{key: value for key, value in self.boundkwargs.items() if key in allowed},
                **kwargs
            })
        finally:
            self.setattr(oldkwargs)

    def __getattribute__(self, name):
        # Needed to avoid binding functions that are saved as properties
        # upon attribute access. Properties should be annotated!
        cls = type(self)
        if name != '__dict__' and name not in self.__dict__ and name in get_type_hints(cls):
            return getattr(cls, name)
        return super().__getattribute__(name)

    plate_stack: Union[Iterable[int], ContextManager] = nullcontext()
    """A stack of plates or an iterable of ints.

       Either one or multiple plates (as returned by `pyro.plate
       <pyro.primitives.plate>` or `pyro.plate_stack
       <pyro.primitives.plate_stack>`) or an iterable of ints that
       will be converted to a stack of plates (named ``plate_0``, etc. and
       aligned to ``rightmost_dim = -1``) for batch mock generation."""

    @property
    def plate(self) -> pyro.plate:
        return (self.plate_stack if isinstance(self.plate_stack, ContextManager)
                else pyro.plate_stack('plate', self.plate_stack))


class SamplingCommand(Command, ABC):
    savename: str = None
    """Filename to save the sample to (or `None` to skip saving)."""

    conditioning: bool = False
    """Whether to retain any conditioning already applied to the model.
       If a false value, `pyro.poutine.handlers.uncondition` will
       be applied to ``model`` before evaluating."""

    @property
    def uncondition(self):
        return pyro.poutine.uncondition() if not self.conditioning else nullcontext()
