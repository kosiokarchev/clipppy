from __future__ import annotations

from functools import partialmethod
from typing import Mapping, Generic, TypeVar, Callable

import attr
from torch import Tensor
from torch.utils._pytree import tree_unflatten
from torch.utils.data.datapipes.iter import IterableWrapper

from .command import LightningSBICommand
from .config import Config
from .loss import BCENRELoss, BaseNRELoss, BaseSBILoss
from ...sbi._typing import SBIBatch, _MultiMappingT, _KT, _SBIParamsT, _SBIObsT
from ...sbi.nn import _HeadOoutT
from ...utils import Sentinel

_BatchT = TypeVar('_BatchT')


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class AbstractNRE(LightningSBICommand[BaseNRELoss, Tensor | _MultiMappingT, _HeadOoutT, _KT], Generic[_BatchT, _HeadOoutT, _KT]):
    loss_config: Config[BaseNRELoss] = attr.ib(factory=lambda: Config(BCENRELoss(), Sentinel.no_call))

    def _loss_one(self, obs: _HeadOoutT, params: _SBIParamsT, decoy: _SBIParamsT, **kwargs):
        return self.lossfunc(self.tail(params, obs, **kwargs), self.tail(decoy, obs, **kwargs))

    def _loss_tree(self, batch: _BatchT, **kwargs) -> tuple[Tensor, tuple[BaseSBILoss.ReturnT, BaseSBILoss.ReturnT]]: ...

    def _step(self, batch: _BatchT, *args, _log_name, **kwargs):
        loss, tree = self._loss_tree(batch)
        self.log_loss(loss, tree, getattr(self, _log_name))
        return loss

    training_step = partialmethod(_step, _log_name='_loss_name')
    validation_step = partialmethod(_step, _log_name='_val_name')


class BasicNRE(AbstractNRE[tuple[SBIBatch, _SBIParamsT],  _HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    def _loss_tree(self, batch: tuple[SBIBatch, _SBIParamsT], **kwargs) -> tuple[Tensor, tuple[BaseSBILoss.ReturnT, BaseSBILoss.ReturnT]]:
        params, obs = self.head(batch[0].params, batch[0].obs)
        ret = self._loss_one(obs, params, batch[1])

        return ret.loss, tree if ret.spec is not None and isinstance(tree := ret.unflatten(), Mapping) else None


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class DynamicNRE(BasicNRE[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    decoy_sampler: Callable[[_SBIObsT], _SBIParamsT] = None

    def _loss_tree(self, batch: SBIBatch, **kwargs):
        return super()._loss_tree((batch, self.decoy_sampler(batch.obs)), **kwargs)


class NRE(AbstractNRE[tuple[SBIBatch, SBIBatch], _HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    def _training_loader(self, dataset):
        return super()._training_loader(IterableWrapper(dataset, deepcopy=False).batch(
            batch_size=2, drop_last=True, wrapper_class=tuple
        ))

    def _loss_comp(self, theta_1: _SBIParamsT, x_1: _HeadOoutT, theta_2: _SBIParamsT, x_2: _HeadOoutT, **kwargs) -> tuple[Tensor, tuple[BaseSBILoss.ReturnT, BaseSBILoss.ReturnT]]:
        ret_1 = self._loss_one(x_1, theta_1, theta_2, **kwargs)
        ret_2 = self._loss_one(x_2, theta_2, theta_1, **kwargs)
        loss = (ret_1.loss + ret_2.loss) / 2
        return loss, (ret_1, ret_2)

    def _loss_tree(self, batch: tuple[SBIBatch, SBIBatch], **kwargs):
        loss, rets = self._loss_comp(
            *self.head(batch[0].params, batch[0].obs),
            *self.head(batch[1].params, batch[1].obs),
            **kwargs
        )

        return loss, tree if rets[0].spec is not None and isinstance(tree := tree_unflatten([
            sum(rs) / len(rets) for rs in zip(*(ret.flat for ret in rets))
        ], rets[0].spec), Mapping) else None
