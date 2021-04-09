import typing as tp

import pyro
import pyro.infer
import pyro.optim
import torch

from .commands import Commandable, Fit, Mock, PPD
from .guide import Guide
from .utils import noop
from .utils.pyro import depoutine
from .utils.typing import _Model


__all__ = 'Clipppy',



class Clipppy(Commandable):
    """
    Attributes
    ----------
    model
        A callable that acts as a generative model.
    guide
        A callable that acts as a guide. Preferably a :any:`Guide`.
    """
    def __init__(self,
                 model: _Model = noop,
                 guide: Guide = Guide(),
                 conditioning: tp.Dict[str, torch.Tensor] = None,
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
