from __future__ import annotations

from typing import Mapping, Union

from .command import LightningSBICommand
from .config import Config
from .loss import NPELoss
from ..npe.nn import NPEResult
from ...sbi._typing import SBIBatch, _KT
from ...sbi.data import GASBIDataset
from ...sbi.nn import BaseGASBITail
from ...utils import Sentinel


_NPEResultT = Union[NPEResult, Mapping[_KT, NPEResult]]


class NPE(LightningSBICommand[_NPEResultT, NPELoss]):
    loss_config: Config[NPELoss] = Config(NPELoss(), Sentinel.no_call)

    def training_step(self, batches: SBIBatch, *args, **kwargs):
        ret = self.lossfunc(self(batches))
        self.log_loss(ret.loss, tree if isinstance(tree := ret.unflatten(), Mapping) else None)
        return ret.loss


class GANPE(NPE):
    dataset_cls = GASBIDataset
    tail: BaseGASBITail

    def training_step(self, batches: SBIBatch, *args, **kwargs):
        # TODO: cringy
        sim_log_prob_grad = self.tail.sim_log_prob_grad(batches[0])
        nperes = self(batches, tail_kwargs=dict(requires_grad=True))
        ret = self.lossfunc(nperes, sim_log_prob_grad)
        self.log_loss(ret.loss, tree if isinstance(tree := ret.unflatten(), Mapping) else None)
        return ret.loss
