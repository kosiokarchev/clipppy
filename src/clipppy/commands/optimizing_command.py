from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any, Callable, Generic, Iterable, Literal, Mapping, Optional, Type, TypeVar, Union

import numpy as np
from more_itertools import always_iterable
from torch import Tensor
from tqdm.auto import trange
from typing_extensions import TypeAlias

from ..commands import Command
from ..utils import merge_if_not_skip, Sentinel


class LossTracker(list[float]):
    def __init__(self, nwindow):
        super().__init__()

        self.nwindow = nwindow

        self._xwindow = np.arange(self.nwindow) - (self.nwindow-1) / 2
        self._xwindow_denom = (self._xwindow**2).sum()

        self.min = float('inf')

    def append(self, val: float):
        super().append(val)

        if val < self.min:
            self.min = val

    @property
    def windowed(self):
        return self[-self.nwindow:]

    @property
    def average(self):
        return np.mean(self.windowed).item()

    @property
    def slope(self):
        if len(self) < self.nwindow:
            return float('nan')
        return (self._xwindow * (self.windowed - np.mean(self._xwindow))).sum() / self._xwindow_denom


_OptimizerT = TypeVar('_OptimizerT')
_LossT = TypeVar('_LossT')
_CallbackT: TypeAlias = Callable[['OptimizingCommand', int, float, tuple[Any], Mapping[str, Any]], Optional[bool]]


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

    callback: Union[_CallbackT, Iterable[_CallbackT]] = None
    """Callback to be executed after each step (or an iterable of such).

       Signature should be ``callback(command, i, loss, kwargs)``, where
          - ``command`` is the `OptimizingCommand` instance
          - ``i`` is the current step index,
          - ``loss`` is the current loss,
          - ``kwargs`` is a dictionary of the parameters passed to the fitting
            function.
       Returning any ``True`` value interrupts the optimization. If multiple
       callbacks are supplied, all of them are still executed, even if some of
       them request interruption."""

    @abstractmethod
    def step(self, *args, **kwargs): ...

    def forward(self, *args, **kwargs):
        losses = LossTracker(self.avgwindow)

        with suppress(KeyboardInterrupt):
            for i in (tq := trange(self.n_steps)):
                loss = self.step(*args, **kwargs)
                losses.append(loss)

                # We don't want to be killed by non-essential stuff
                try:
                    # TODO: self.lr need not reflect the true lr
                    slope = losses.slope / self.lr

                    tq.set_postfix_str(f'loss={loss:_.3f} (avg={losses.average:_.3f}, min={losses.min:_.3f}, slope={slope:.3e})')
                except KeyboardInterrupt as e:
                    raise e from None
                except Exception as e:
                    # TODO: report those errors
                    pass

                if (
                    any([callback(self, i, loss, args, kwargs) for callback in always_iterable(self.callback)])
                    or (0 < self.min_steps <= i and self.converged(slope, losses.windowed))
                ):
                    break

        return losses
