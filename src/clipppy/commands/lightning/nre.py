from __future__ import annotations

from typing import Mapping

from torch import Tensor
from torch.utils._pytree import tree_unflatten

from .command import LightningSBICommand
from .config import Config
from .loss import NRELoss
from ...sbi._typing import _SBIBatchT, DEFAULT_LOSS_NAME
from ...sbi.data import DoublePipe
from ...utils import Sentinel


class NRE(LightningSBICommand[Tensor, NRELoss]):
    @property
    def training_loader(self):
        return self.loader_config(DoublePipe(self.dataset))

    loss_config: Config[NRELoss] = Config(NRELoss(), Sentinel.no_call)

    def training_step(self, batches: tuple[_SBIBatchT, _SBIBatchT], *args, **kwargs):
        theta_1, x_1 = self.head(*batches[0])
        theta_2, x_2 = self.head(*batches[1])

        ret_1 = self.lossfunc(self.tail(theta_1, x_1), self.tail(theta_2, x_1))
        ret_2 = self.lossfunc(self.tail(theta_2, x_2), self.tail(theta_1, x_2))

        loss = (ret_1.loss + ret_2.loss) / 2

        # noinspection PyUnboundLocalVariable
        self.log_loss(loss, tree if ret_1.spec is not None and isinstance(tree := tree_unflatten([
            (a + b) / 2 for a, b in zip(ret_1.flat, ret_2.flat)
        ], ret_1.spec), Mapping) else None)

        return loss
