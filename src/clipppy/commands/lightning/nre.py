from __future__ import annotations

from functools import partialmethod
from typing import Mapping

from torch import Tensor
from torch.utils._pytree import tree_unflatten

from .command import LightningSBICommand
from .config import Config
from .loss import NRELoss
from ...sbi._typing import SBIBatch
from ...sbi.data import DoublePipe
from ...utils import Sentinel


class NRE(LightningSBICommand[Tensor, NRELoss]):
    def _training_loader(self, dataset):
        return super()._training_loader(DoublePipe(dataset))

    loss_config: Config[NRELoss] = Config(NRELoss(), Sentinel.no_call)

    def _loss_tree(self, batches: tuple[SBIBatch, SBIBatch]):
        theta_1, x_1 = self.head(batches[0].params, batches[0].obs)
        theta_2, x_2 = self.head(batches[1].params, batches[1].obs)

        ret_1 = self.lossfunc(
            self.tail(theta_1, x_1), self.tail(theta_2, x_1),
            # batches[0].weight, batches[0].weight
        )
        ret_2 = self.lossfunc(
            self.tail(theta_2, x_2), self.tail(theta_1, x_2),
            # batches[1].weight, batches[1].weight
        )

        loss = (ret_1.loss + ret_2.loss) / 2

        return loss, tree if ret_1.spec is not None and isinstance(tree := tree_unflatten([
            (a + b) / 2 for a, b in zip(ret_1.flat, ret_2.flat)
        ], ret_1.spec), Mapping) else None

    def _step(self, batches: tuple[SBIBatch, SBIBatch], *args, _log_name, **kwargs):
        loss, tree = self._loss_tree(batches)
        self.log_loss(loss, tree, getattr(self, _log_name))
        return loss

    training_step = partialmethod(_step, _log_name='_loss_name')
    validation_step = partialmethod(_step, _log_name='_val_name')
