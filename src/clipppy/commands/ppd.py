from __future__ import annotations

from typing import TypedDict

import pyro
import pyro.poutine
import torch

from .command import SamplingCommand
from ..guide import Guide
from ..utils.typing import _Model


class PPD(SamplingCommand):
    """Sample from the guide and optionally generate the corresponding data."""

    observations: bool = True
    """Sample also the observations corresponding to the drawn parameters."""

    guidefile: str = None
    """File to load a guide from, or `None` to use the guide in the config."""

    def forward(self, model: _Model, guide: Guide, *args, **kwargs)\
            -> TypedDict('ppd', {'guide_trace': pyro.poutine.Trace,
                                 'model_trace': pyro.poutine.Trace}, total=False):
        # TODO: better guide loading
        if self.guidefile is not None:
            guide = torch.load(self.guidefile)

        guide_is_trainable = hasattr(guide, 'training')

        if guide_is_trainable:
            was_training = guide.training
            guide.eval()
        with pyro.poutine.trace() as guide_tracer, self.plate:
            guide(*args, **kwargs)
        if guide_is_trainable:
            guide.train(was_training)
        ret = {'guide_trace': guide_tracer.trace}

        if self.observations:
            with pyro.poutine.trace() as model_tracer, self.plate,\
                    pyro.poutine.replay(trace=ret['guide_trace']), self.uncondition:
                model(*args, **kwargs)
            ret['model_trace'] = model_tracer.trace

        if self.savename:
            torch.save(ret, self.savename)

        return ret
