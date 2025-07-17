from __future__ import annotations

from typing import Mapping, TypedDict

from pyro import condition
from torch import Tensor
from typing_extensions import Unpack

# noinspection PyCompatibility
from . import commands
from .commands.commandable import Commandable
from .guide.guide import Guide
from .utils import noop
from .utils.pyro import depoutine
from .utils.typing import _Model


__all__ = 'Clipppy',


class Clipppy(Commandable):
    class _KwargsT(TypedDict, total=False):
        fit: commands.Fit._KwargsT
        mock: commands.Mock._KwargsT
        ppd: commands.PPD._KwargsT
        lightning_npe: commands.LightningNPE._KwargsT
        lightning_nre: commands.LightningNRE._KwargsT

    def __init__(self,
                 model: _Model = noop,
                 guide: Guide = Guide(),
                 conditioning: Mapping[str, Tensor] = None,
                 **kwargs: Unpack[Clipppy._KwargsT]):
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
        return condition(self._model, data=self.conditioning)

    mock: commands.Mock

    fit: commands.Fit
    ppd: commands.PPD

    npe: commands.NPE

    nre: commands.NRE
