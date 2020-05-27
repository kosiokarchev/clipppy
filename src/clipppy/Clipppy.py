import typing as tp
from contextlib import nullcontext
from itertools import chain

import numpy as np
import pyro
import pyro.infer
import pyro.optim
import torch
from pyro.infer import Trace_ELBO, SVI
from tqdm import tqdm

from .globals import register_globals, init_msgr
from .guide import Guide


__all__ = ('Clipppy',)


class Clipppy:
    """
    Attributes
    ----------
    model
        A callable that acts as a geerative model.
    guide : Guide
        A callable that acts as a guide. Preferably a :any:`Guide`.
    """
    def __init__(self, model, guide: Guide, conditioning: tp.Dict[str, torch.Tensor] = None):
        # Conditions the model and sets it on the guide, if it doesn't have a model already.
        self.conditioning = conditioning
        self.model = model if conditioning is None else pyro.condition(model, data=conditioning)
        self.guide = guide

        if isinstance(self.guide, Guide) and self.guide.model is None:
            self.guide.model = self.model

    def fit(self, n_steps, lr=1e-3, min_steps=1000, n_write=300, num_particles=1, conv_th=0.0,
            optimizer_cls: tp.Union[tp.Callable[[tp.Dict], pyro.optim.PyroOptim], tp.Type[pyro.optim.PyroOptim]] = pyro.optim.Adam,
            optimizer_args=None,
            avgwindow=100,
            *args, **kwargs):
        """Fit using ELBO maximisation.

        Parameters
        ----------
        n_steps,min_steps,n_write
            Run at least ``min_steps`` and at most ``n_steps``, saving the guide
            each ``n_write`` steps.
        conv_th
            If the average decrease in the loss over ``avgwindow`` steps falls
            below ``conv_th * lr``, stop the fit.
        avgwindow
            Number of most recents steps to use for calculating various
            averaged quantities (average rate, mean loss...).
        lr
            Learning rate (passed to the optimizer).
        optimizer_cls
            The class of `PyroOptim <pyro.optim.optim.PyroOptim>` to
            instantiate (or a function that acts like it).
        optimizer_args
            The (keyword!) arguments to pass to ``optimizer_cls``. Will be
            updated with the standalone ``lr`` passed.
        num_particles
            Number of particles to use.
        *args,**kwargs
            Passed on to the `model` and `guide`.
        """
        optimizer_args = dict(
            chain(
                {'amsgrad': False, 'weight_decay': 0.}.items(),
                optimizer_args.items() if optimizer_args is not None else ()
            ),
            lr=lr
        )

        optimizer = optimizer_cls(optimizer_args)
        lossfunc = Trace_ELBO(num_particles=num_particles)

        svi = SVI(self.model, self.guide, optimizer, lossfunc)

        xavg = np.arange(avgwindow)
        slope = np.NaN
        losses = []
        minloss = None
        with tqdm(range(n_steps)) as tq:
            for i in tq:
                loss = svi.step(*args, **kwargs)
                losses.append(loss)

                windowed_loss = losses[-avgwindow:]
                avgloss = np.mean(windowed_loss)
                minloss = min(minloss, loss) if minloss is not None else loss

                if len(windowed_loss) >= avgwindow:
                    slope = np.polyfit(xavg, windowed_loss, deg=1)[0] / lr

                tq.set_postfix_str(f'loss={loss:.3f} (avg={avgloss:.3f}) (min={minloss:.3f}) (slope={slope:.3e})')

    def mock(self, plate_stack: tp.Union[tp.Iterable[int], tp.ContextManager] = nullcontext(),
             conditioning: bool = False,
             *args, **kwargs) -> pyro.poutine.Trace:
        """Generate mock data.

        Parameters
        ----------
        plate_stack
            Either one or multiple plates (as returned by `pyro.plate
            <pyro.primitives.plate>` or `pyro.plate_stack
            <pyro.primitives.plate_stack>`) or an iterable of ints that
            will be converted to a stack of plates (named ``plate_0``, etc. and
            aligned to ``rightmost_dim = -1``) for batch mock generation.
        conditioning
            Whether to retain any conditioning already applied to the model.
            If a `False` value, `pyro.poutine.handlers.uncondition` will
            be applied to `model` before evaluating.
        *args,**kwargs
            Passed on to `model`.

        Returns
        -------
        `pyro.poutine.Trace`

        """
        plate_stack = (plate_stack if isinstance(plate_stack, tp.ContextManager)
                       else pyro.plate_stack('plate', plate_stack))
        uncondition = pyro.poutine.uncondition() if not conditioning else nullcontext()
        with pyro.poutine.trace() as trace, init_msgr, plate_stack, uncondition:
            self.model(*args, **kwargs)

        return trace.trace


register_globals(**{a: globals()[a] for a in __all__ if a in globals()})
