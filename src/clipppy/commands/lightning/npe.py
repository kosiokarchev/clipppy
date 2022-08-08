from __future__ import annotations

from typing import Mapping

from torch.utils._pytree import tree_unflatten

from .command import LightningSBICommand
from .config import Config
from .loss import MultiNPELoss
from ..npe.nn import BaseNPETail, NPEResult
from ..sbi._typing import _SBIBatchT
from ...utils import Sentinel


class NPE(LightningSBICommand[NPEResult, MultiNPELoss]):
    loss_config: Config[MultiNPELoss] = Config(MultiNPELoss(), Sentinel.no_call)

    def training_step(self, batches: _SBIBatchT, *args, **kwargs):
        ret = self.lossfunc(self(batches))
        self.log_loss(ret.loss, tree if ret.spec is not None and isinstance(tree := tree_unflatten(ret.flat, ret.spec), Mapping) else None)
        return ret.loss
