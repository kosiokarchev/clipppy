from __future__ import annotations

from functools import partialmethod
from typing import Mapping, TYPE_CHECKING, Union, Generic

from torch import Tensor
from torch.utils._pytree import tree_unflatten
from torchdata.datapipes.iter import IterableWrapper
from typing_extensions import Unpack

from .command import LightningSBICommand
from .config import Config
from .loss import BCENRELoss, BaseNRELoss, BaseSBILoss
from ...sbi._typing import SBIBatch, _MultiMappingT, _KT
from ...sbi.nn import _HeadOoutT
from ...utils import Sentinel


class NRE(LightningSBICommand[BaseNRELoss, Union[Tensor, _MultiMappingT], _HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    class _KwargsT(LightningSBICommand._KwargsT, total=False):
        loss_config: Config[BaseNRELoss]

    if TYPE_CHECKING:
        # noinspection PyMissingConstructor
        def __init__(self, **kwargs: Unpack[_KwargsT]): ...

    def _training_loader(self, dataset):
        return super()._training_loader(IterableWrapper(dataset, deepcopy=False).batch(
            batch_size=2, drop_last=True, wrapper_class=tuple
        ))

    loss_config: Config[BaseNRELoss] = Config(BCENRELoss(), Sentinel.no_call)

    def _loss_one(self, obs, params, decoy, **kwargs):
        return self.lossfunc(self.tail(params, obs, **kwargs), self.tail(decoy, obs, **kwargs))

    def _loss_comp(self, theta_1: Tensor, x_1: Tensor, theta_2: Tensor, x_2: Tensor, **kwargs) -> tuple[Tensor, tuple[BaseSBILoss.ReturnT, ...]]:
        ret_1 = self._loss_one(x_1, theta_1, theta_2, **kwargs)
        ret_2 = self._loss_one(x_2, theta_2, theta_1, **kwargs)
        loss = (ret_1.loss + ret_2.loss) / 2
        return loss, (ret_1, ret_2)

    def _loss_tree(self, batches: tuple[SBIBatch, SBIBatch], **kwargs):
        loss, rets = self._loss_comp(
            *self.head(batches[0].params, batches[0].obs),
            *self.head(batches[1].params, batches[1].obs),
            **kwargs
        )

        return loss, tree if rets[0].spec is not None and isinstance(tree := tree_unflatten([
            sum(rs) / len(rets) for rs in zip(*(ret.flat for ret in rets))
        ], rets[0].spec), Mapping) else None

    def _step(self, batches: tuple[SBIBatch, SBIBatch], *args, _log_name, **kwargs):
        loss, tree = self._loss_tree(batches)
        self.log_loss(loss, tree, getattr(self, _log_name))
        return loss

    training_step = partialmethod(_step, _log_name='_loss_name')
    validation_step = partialmethod(_step, _log_name='_val_name')
