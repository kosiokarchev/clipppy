from __future__ import annotations

from functools import partialmethod
from typing import Mapping, Union, Generic

from .command import LightningSBICommand
from .config import Config
from .loss import NPELoss
from ...sbi._typing import SBIBatch, _KT, _MultiKT
from ...sbi.nn import _HeadOoutT
from ...sbi.nn.npe import NPEResult
from ...utils import Sentinel


class NPE(LightningSBICommand[NPELoss, Union[NPEResult, Mapping[_MultiKT, NPEResult]], _HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    class _KwargsT(LightningSBICommand._KwargsT):
        loss_config: Config[NPELoss]

    loss_config: Config[NPELoss] = Config(NPELoss(), Sentinel.no_call)

    def _step(self, batch: SBIBatch, *args, _log_name, **kwargs):
        ret = self.lossfunc(self(batch))
        self.log_loss(
            ret.loss, tree if isinstance(tree := ret.unflatten(), Mapping) else None,
            loss_name=getattr(self, _log_name)
        )
        return ret.loss

    training_step = partialmethod(_step, _log_name='_loss_name')
    validation_step = partialmethod(_step, _log_name='_val_name')
