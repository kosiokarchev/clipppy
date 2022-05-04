from __future__ import annotations

from pyro.infer import ELBO, SVI, Trace_ELBO
from pyro.optim import PyroOptim, Adam

from .optimizing_command import OptimizingCommand
from ..utils.typing import _Guide, _Model


class Fit(OptimizingCommand[PyroOptim, ELBO]):
    """Fit using ELBO maximisation."""

    n_write: int = -1
    """Save the guide each ``n_write`` steps."""

    optimizer_cls = Adam

    def _instantiate_optimizer(self, kwargs):
        return self.optimizer_cls(kwargs)

    loss_cls = Trace_ELBO

    def step(self, svi: SVI, /, *args, **kwargs):
        return svi.step(*args, **kwargs)

    def forward(self, model: _Model, guide: _Guide, *args, **kwargs):
        return super().forward(
            SVI(model, guide, self.optimizer, self.lossfunc),
            *args, **kwargs)
