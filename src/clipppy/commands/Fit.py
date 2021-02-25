import typing as tp

import numpy as np
import pyro.infer
import pyro.optim
import torch
from pyro.infer import SVI, Trace_ELBO
from tqdm.auto import tqdm

from .Command import Command
from ..utils import dict_union, noop
from ..utils.typing import _Guide, _Model


class Fit(Command):
    """Fit using ELBO maximisation."""

    n_steps: int = 1000
    """Run at most ``n_steps`` steps."""

    min_steps: int = 1
    """Run at least ``min_steps`` steps."""

    n_write: int = -1
    """Save the guide each ``n_write`` steps."""

    lr: float = 1e-3
    """Learning rate (passed to the optimizer)."""

    conv_th: float = -float('inf')
    """Convergence threshold. Should be positive or ``-float('inf')`` to turn
       off convergence-based termination.

       See `converged`."""

    avgwindow: int = 100
    """Number of most recents steps to use for calculating various
       averaged quantities (average rate, mean loss...)."""

    optimizer_cls: tp.Union[tp.Type[pyro.optim.PyroOptim],
                            tp.Callable[[tp.Dict], pyro.optim.PyroOptim]]\
        = pyro.optim.Adam
    """The class of `PyroOptim <pyro.optim.optim.PyroOptim>` to
       instantiate (or a function that acts like it)."""

    optimizer_args: tp.Optional[tp.Dict] = {}
    """The (keyword!) arguments to pass to ``optimizer_cls``.

       Will be updated with the standalone ``lr`` passed, unless it is ``noop``.

       Pass the special value `Command.no_call` to avoid instantiating
       ``optimizer_cls`` and use it directly."""

    loss_cls: tp.Union[tp.Type[pyro.infer.ELBO], tp.Callable[..., torch.Tensor]]\
        = Trace_ELBO
    """The loss class to instantiate."""

    loss_args: tp.Dict = {}
    """Arguments for the ``loss_cls`` constructor.

       Pass the special value `Command.no_call` to avoid instantiating
       ``load_cls`` and use it directly."""

    callback: tp.Callable[[int, float, tp.Mapping[str, tp.Any]], tp.Any] = None
    """Callback to be executed after each step.

       Signature should be ``callback(i, loss, locals)``, where ``i`` is the
       current step index, ``loss`` is the current loss, and ``locals`` is a
       dictionary of the local variables in the fitting function. Depending on
       the python implementation, modifying its values might influence the
       fitting. Returning any ``True`` value interrupts the fitting."""

    @property
    def optimizer(self) -> pyro.optim.PyroOptim:
        """
        Construct an optimizer as ``optimizer_cls(optimizer_args)`` or simply
        return `optimizer_cls` if `optimizer_args` is `Command.no_call`.
        """
        return (self.optimizer_cls if self.optimizer_args is Command.no_call
                else self.optimizer_cls(
            self.optimizer_args if self.lr is noop else
            dict_union(self.optimizer_args, {'lr': self.lr})))

    @property
    def lossfunc(self) -> tp.Union[pyro.infer.ELBO, tp.Callable[..., torch.Tensor]]:
        """
        Construct a loss as ``loss_cls(**loss_args)`` or simply return
        `loss_cls` if `loss_args` is `Command.no_call`.
        """
        return (self.loss_cls if self.loss_args is self.no_call
                else self.loss_cls(**self.loss_args))

    def converged(self, slope: float, windowed_losses: tp.Iterable[float] = None):
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

    def forward(self, model: _Model, guide: _Guide, *args, **kwargs):
        svi = SVI(model, guide, self.optimizer, self.lossfunc)

        xavg = np.arange(self.avgwindow)
        slope = np.NaN
        losses = []
        minloss = None
        try:
            with tqdm(range(self.n_steps)) as tq:
                for i in tq:
                    loss = svi.step(*args, **kwargs)
                    losses.append(loss)

                    windowed_losses = losses[-self.avgwindow:]
                    avgloss = np.mean(windowed_losses)
                    minloss = min(minloss, loss) if minloss is not None else loss

                    if len(windowed_losses) >= self.avgwindow:
                        slope = np.polyfit(xavg, windowed_losses, deg=1)[0] / self.lr

                    tq.set_postfix_str(f'loss={loss:.3f} (avg={avgloss:.3f}, min={minloss:.3f}, slope={slope:.3e})')

                    if (callable(self.callback) and self.callback(i, loss, locals())) or (0 < self.min_steps <= i and self.converged(slope, windowed_losses)):
                        break
        except KeyboardInterrupt:
            pass

        return losses
