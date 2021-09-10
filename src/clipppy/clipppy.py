from __future__ import annotations

from typing import Mapping

import pyro
import pyro.infer
import pyro.optim
import torch

from .commands.commandable import Commandable
from .commands.fit import Fit
from .commands.mock import Mock
from .commands.ppd import PPD
from .guide.guide import Guide
from .utils import noop
from .utils.pyro import depoutine
from .utils.typing import _Model


__all__ = 'Clipppy',


class Clipppy(Commandable):
    def __init__(self,
                 model: _Model = noop,
                 guide: Guide = Guide(),
                 conditioning: Mapping[str, torch.Tensor] = None,
                 **kwargs):
        # Conditions the model and sets it on the guide, if it doesn't have a model already.
        self.conditioning = conditioning if conditioning is not None else {}
        self._model = model
        self.guide = guide

        if isinstance(self.guide, Guide) and self.guide.model is None:
            self.guide.model = self.model

        self.kwargs = {}
        for key, val in kwargs.items():
            if self.get_cmd_cls(key) is not None:
                setattr(self, key, val)
            else:
                self.kwargs[key] = val

    @property
    def umodel(self):
        return depoutine(self._model)

    @property
    def model(self):
        return pyro.condition(self._model, data=self.conditioning)

    fit: Fit
    mock: Mock
    ppd: PPD
