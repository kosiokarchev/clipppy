from __future__ import annotations

from abc import abstractmethod, ABC
from math import log
from typing import Iterable, Mapping, TypeVar, Generic
from warnings import catch_warnings, filterwarnings

import attr
import torch
from torch import Tensor

from ._typing import _MultiKT, SBIBatch, BaseMultiSBIResultRep, _MultiMappingT, MultiNREProtocol, MultiSBIProtocol


_SBIT = TypeVar('_SBIT', bound=MultiSBIProtocol)


@attr.define
class MultiSBIValidator(BaseMultiSBIResultRep, Generic[_SBIT], ABC):
    progress: bool = True

    @staticmethod
    def _credibility(log_weights: _MultiMappingT, log_ratios: _MultiMappingT, ref_log_ratios: _MultiMappingT):
        with catch_warnings():
            filterwarnings('ignore', module='torch.masked', category=UserWarning)
            return {
                key: torch.masked.masked_tensor(
                    log_weights[key].exp(),
                    ref_log_ratios[key] > log_ratios[key].unsqueeze(-1)
                ).sum(-1).to_tensor(0)
                for key in log_ratios.keys() & ref_log_ratios.keys()
            }

    @abstractmethod
    def eval(self, groups: Iterable[_MultiKT], net: _SBIT, batch: SBIBatch) -> tuple[_MultiMappingT, _MultiMappingT]: ...

    def credibility(self, groups: Iterable[_MultiKT], net: _SBIT, batch: SBIBatch):
        log_ratios, ref_log_ratios = self.eval(groups, net, batch)
        return self._credibility({
            key: val-val.logsumexp(-1, keepdim=True) for key, val in ref_log_ratios.items()
        }, log_ratios, ref_log_ratios)

    def validate_batch(self, groups: Iterable[_MultiKT], net: _SBIT, batch: SBIBatch) -> tuple[Mapping[_MultiKT, Tensor], Mapping[_MultiKT, Tensor], Mapping[_MultiKT, Tensor]]:
        log_ratios, ref_log_ratios = self.eval(groups, net, batch)
        log_norms_param, log_norms_data, log_weights = map(dict, zip(*(
            (
                (key, logsum_param.squeeze(-1) - log(val.shape[-1])),
                (key, val.logsumexp(-2) - log(val.shape[-2])),
                (key, val - logsum_param))
            for key, val in ref_log_ratios.items()
            for logsum_param in [val.logsumexp(-1, keepdim=True)]
        )))
        return log_norms_param, log_norms_data, self._credibility(log_weights, log_ratios, ref_log_ratios)

    def validate(self, groups: Iterable[_MultiKT], net: _SBIT, dataset: Iterable[SBIBatch]) -> tuple[Mapping[_MultiKT, Tensor], Mapping[_MultiKT, Tensor], Mapping[_MultiKT, Tensor]]:
        from tqdm.auto import tqdm

        return tuple({
            key: torch.cat([r[key] for r in res]) for key in groups
        } for res in zip(*(self.validate_batch(groups, net, batch) for batch in (
            tqdm(dataset, leave=False, desc='P--P validation')
            if self.progress else dataset
        ))))

    @staticmethod
    def subtype(protocol_type: type[MultiSBIProtocol]) -> type[MultiSBIValidator]:
        return {
            MultiNREProtocol: MultiNREValidator
        }[protocol_type]


class MultiNREValidator(MultiSBIValidator[MultiNREProtocol]):
    def eval(self, groups: Iterable[_MultiKT], net: MultiNREProtocol, batch: SBIBatch):
        return (
            self._eval_nre(groups, net, batch.params, batch.obs),
            {key: val.T for key, val in self._eval_nre(
                groups, net, {key: self._samples[key].unsqueeze(1) for key in net.param_names}, batch.obs
            ).items()}
            # self._eval_nre(groups, net, self._samples, {
            #     key: obs.unsqueeze(-1-net.head.event_dims.get(key, 0))
            #     for key, obs in batch.obs.items()})
        )
