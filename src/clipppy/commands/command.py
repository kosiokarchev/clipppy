from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from contextlib import nullcontext, suppress
from functools import lru_cache
from typing import (
    Any, Callable, ContextManager, Generic, get_type_hints, Iterable,
    Literal, Mapping, Type, TypeVar, Union
)

import numpy as np
import pyro
import pyro.optim
from torch import Tensor
from tqdm.auto import trange

from . import commandable
from ..utils import merge_if_not_skip, Sentinel
from ..utils.pyro import init_msgr


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

    commander: commandable.Commandable

    boundkwargs: dict
    """A dictionary of values to be forwarded to each call of `forward`,
       if there are exact name matches in its signature."""

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
        try:
            kwargs = self.setattr(kwargs)
            allowed = inspect.signature(self.forward).parameters
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
                else pyro.plate_stack('plate', self.plate_stack) if self.plate_stack
                else nullcontext())


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

    initting: bool = True
    """Whether to respect ``init`` values in config sites."""

    @property
    def init(self):
        return init_msgr if self.initting else nullcontext()


_OptimizerT = TypeVar('_OptimizerT')
_LossT = TypeVar('_LossT')


class OptimizingCommand(Command, Generic[_OptimizerT, _LossT], ABC):
    n_steps: int = 1000
    """Run at most ``n_steps`` steps."""

    min_steps: int = 1
    """Run at least ``min_steps`` steps."""

    avgwindow: int = 100
    """Number of most recents steps to use for calculating various
       averaged quantities (average rate, mean loss...)."""

    conv_th: float = -float('inf')
    """Convergence threshold. Should be positive or ``-float('inf')`` to turn
       off convergence-based termination.

       See `converged`."""

    def converged(self, slope: float, windowed_losses: Iterable[float] = None):
        """
        Indicate whether the fit is considered converged.

        This implementation returns `True` if the averaged slope ``slope``,
        defined as the fitted slope of losses over the last `avgwindow`
        iterations, divided by the learning rate, is shallower than `conv_th`,
        which should be given as positive. If the ``slope`` is positive, this
        will always be true, so turning this check off requires setting
        ``conv_th = -float('inf')``.

        This method can be overridden in subclasses to provide more
        sophisticated checks.

        """
        return slope > -self.conv_th

    lr: Union[float, Literal[Sentinel.skip]] = 1e-3
    """Learning rate (passed to the optimizer)."""

    @staticmethod
    def _instantiate(cls, kwargs, add_kwargs, instantiator=None):
        if instantiator is None:
            instantiator = lambda kw: cls(**kw)

        return (cls if kwargs is Sentinel.no_call
                else instantiator(merge_if_not_skip(kwargs, add_kwargs)))

    optimizer_cls: Union[Type[_OptimizerT], Callable[..., _OptimizerT], _OptimizerT]
    """The class of optimizer to instantiate (or a function that acts like it);
       or a ready optimizer if ``optimizer_args`` is `Sentinel.no_call`."""

    optimizer_args: Union[Mapping, Literal[Sentinel.skip]] = {}
    """The (keyword!) arguments to pass to ``optimizer_cls``.

       Will be updated with the standalone ``lr`` passed, unless it is ``noop``.

       Pass the special value `Sentinel.no_call` to avoid instantiating
       ``optimizer_cls`` and use it directly."""

    def _instantiate_optimizer(self, kwargs) -> _OptimizerT:
        return self.optimizer_cls(**kwargs)

    @property
    def optimizer(self) -> _OptimizerT:
        """
        Construct an optimizer as ``optimizer_cls(**optimizer_args)`` or simply
        return `optimizer_cls` if `optimizer_args` is `Sentinel.no_call`.
        """
        return self._instantiate(self.optimizer_cls, self.optimizer_args, {'lr': self.lr}, self._instantiate_optimizer)

    loss_cls: Union[Type[_LossT], Callable[..., _LossT], _LossT]
    """The loss class to instantiate (or a function that acts like it);
       or a ready loss function if ``loss_args`` is `Sentinel.no_call`."""

    loss_args: Union[Mapping, Literal[Sentinel.skip]] = {}
    """The (keyword!) arguments to pass to ``loss_cls``.

       Pass the special value `Sentinel.no_call` to avoid instantiating
       ``load_cls`` and use it directly."""

    @property
    def lossfunc(self) -> Union[_LossT, Callable[..., Tensor]]:
        """
        Construct a loss as ``loss_cls(**loss_args)`` or simply return
        `loss_cls` if `loss_args` is `Sentinel.no_call`.
        """
        return self._instantiate(self.loss_cls, self.loss_args, {})

    callback: Callable[[int, float, Mapping[str, Any]], Any] = None
    """Callback to be executed after each step.

       Signature should be ``callback(i, loss, locals)``, where ``i`` is the
       current step index, ``loss`` is the current loss, and ``locals`` is a
       dictionary of the local variables in the fitting function. Depending on
       the python implementation, modifying its values might influence the
       fitting. Returning any ``True`` value interrupts the fitting."""

    @abstractmethod
    def step(self, *args, **kwargs): ...

    def forward(self, *args, **kwargs):
        xavg = np.arange(self.avgwindow)
        slope = np.NaN
        losses = []
        minloss = None

        with suppress(KeyboardInterrupt):
            for i in (tq := trange(self.n_steps)):
                loss = self.step(*args, **kwargs)
                losses.append(loss)

                # We don't want to be killed by non-essential stuff
                try:
                    windowed_losses = losses[-self.avgwindow:]
                    avgloss = np.mean(windowed_losses)
                    minloss = min(minloss, loss) if minloss is not None else loss

                    if len(windowed_losses) >= self.avgwindow:
                        slope = np.polyfit(xavg, windowed_losses, deg=1)[0] / self.lr

                    tq.set_postfix_str(f'loss={loss:_.3f} (avg={avgloss:_.3f}, min={minloss:_.3f}, slope={slope:.3e})')
                except KeyboardInterrupt as e:
                    raise e from None
                except Exception as e:
                    # TODO: report those errors
                    pass

                if (callable(self.callback) and self.callback(i, loss, locals())) or (0 < self.min_steps <= i and self.converged(slope, windowed_losses)):
                    break

        return losses
