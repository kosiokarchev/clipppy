from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Iterable, Mapping, Protocol, TypeVar, Generic, Any

import attr
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.utils.data import default_collate
from tqdm.auto import tqdm
from typing_extensions import TypeAlias

from phytorchx.dataframe import TensorDataFrame

from ..utils.typing import _KT

_MultiKT = _KT | Iterable[_KT]
_MultiMappingT: TypeAlias = Mapping[_MultiKT, Tensor]


_TreeV = TypeVar('_TreeV')
_Tree: TypeAlias = _TreeV | Iterable['_Tree'] | Mapping[Any, '_Tree']
_SBIParamsT: TypeAlias = Mapping[_KT, Tensor]
_SBIObsT: TypeAlias = Mapping[_KT, Tensor]
_SBIWeightT: TypeAlias = _Tree[Tensor | Number]


@dataclass
class SBIBatch(Generic[_KT]):
    params: _SBIParamsT
    obs: _SBIObsT
    weight: _SBIWeightT = 1


# Here because of circular import
from . import data, nn


class MultiSBIProtocol(Protocol):
    param_names: Iterable[str]
    obs_names: Iterable[str]

    loader: data.SBIDataLoader
    dataset: data.SBIDataset

    head: nn.BaseSBIHead
    tail: nn.BaseMultiSBITail

    @staticmethod
    def resolve(sbi_type) -> type[MultiSBIProtocol]:
        from ..commands.lightning.nre import AbstractNRE
        if issubclass(sbi_type, AbstractNRE):
            return MultiNREProtocol

        from ..commands.lightning.npe import NPE
        if issubclass(sbi_type, NPE):
            return MultiNPEProtocol

        raise TypeError


class MultiNREProtocol(MultiSBIProtocol):
    def __call__(self, batch: SBIBatch, **kwargs) -> Mapping[_MultiKT, Tensor]: ...


class MultiNPEProtocol(MultiSBIProtocol):
    def posterior(self, obs: _SBIObsT) -> Mapping[_MultiKT, Distribution]: ...


@attr.define(slots=False)
class BaseMultiSBIResultRep:
    _samples: Mapping[_KT, Tensor]

    def to(self, device: str | torch.device = None):
        self._samples = {key: val.to(device) for key, val in self._samples.items()}
        return self

    batch_size: int = attr.ib(default=None, kw_only=True)
    batched_progress: bool = attr.ib(default=True, kw_only=True)

    def _batched_iter(self, params: _SBIParamsT):
        params = TensorDataFrame(params)
        ret = params.batched(self.batch_size or len(params), shuffle=False)
        return tqdm(ret, leave=False) if self.batched_progress and self.batch_size else ret

    def _eval(self, groups, net, params, obs, post) -> _MultiMappingT:
        from clipppy.sbi.nn import MultiSBITail

        with torch.inference_mode():
            res = [
                {group: post(net.tail.forward_one(group, *headout)) for group in groups}
                for batch in self._batched_iter(params)
                for headout in [net.head(batch, obs)]
            ] if isinstance(net.tail, MultiSBITail) else [
                post(net.tail(*net.head(batch, obs)))
                for batch in self._batched_iter(params)
            ]

        return {key: val.flatten(end_dim=1) for key, val in default_collate(res).items()}

    def _eval_nre(self, groups: Iterable[_MultiKT], net: MultiNREProtocol, params: _SBIParamsT, obs: _SBIObsT) -> _MultiMappingT:
        return self._eval(groups, net, params, obs, lambda x: x)


_OptimizerT = TypeVar('_OptimizerT', bound=Optimizer)
_SchedulerT = TypeVar('_SchedulerT')
DEFAULT_LOSS_NAME = 'loss'
DEFAULT_VAL_NAME = 'val'
