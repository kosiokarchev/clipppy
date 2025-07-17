from abc import ABC, abstractmethod
from typing import Union, Callable, Sequence, Generic, Mapping, cast, Optional

import attr
import torch
from more_itertools import unique_everseen, one
from torch import Tensor, LongTensor
from torch.nn import Module

from phytorchx import broadcast_cat, broadcast_gather
from . import SBIHead, _HeadOoutT
from .nre import ParamPackerNRETail, NRETail
from .._typing import _KT, _SBIParamsT
from ...utils.nn import LazyWhitenOnline, WhitenOnline
from ...utils.nn.empty import _empty_module
from ...utils.nn.sets import BatchedSetModule, SetBatch, SetNorm, CrossEncoder


@attr.s(eq=False, auto_attribs=True)
class SetSBIMixin:
    set_dim: int = attr.ib(default=0, kw_only=True)

    def _nested_cat(self, nt: Sequence[Tensor]):
        return torch.cat(tuple(t.movedim(self.set_dim, 0) for t in nt), 0)
        # return torch.Tensor(nt.storage()).reshape(-1, *map(nt.size, range(2, nt.ndim)))


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class SubsetSBIMixin(SetSBIMixin):
    subsample: int = None
    subsampling: bool = True

    def _subsample(self, params: _SBIParamsT, obs: SetBatch) -> tuple[_SBIParamsT, Tensor]:
        if self.subsampling and self.subsample is not None:
            idx = (torch.rand((len(obs.sizes), self.subsample), device=obs.val.device) * obs.sizes.unsqueeze(-1)).long()
            obs_out = broadcast_gather(obs.val, 0, obs.indptr[:-1].unsqueeze(-1) + idx, 2)

            params_out = {key: (
                torch.atleast_1d(val).unsqueeze(1).expand(val.shape[0], self.subsample, *val.shape[1:])
                if isinstance(val, Tensor) else
                torch.stack([(
                    v[ind] if len(v) == s.item() else
                    # NB: length might not match in marginal pairs, so:
                    v[torch.randint(len(v), (self.subsample,), device=v.device)]
                ) for v, ind, s in zip(val, idx, obs.sizes)], 0)
            ) for key, val in params.items()}
        else:
            assert len(obs.sizes) == 1
            params_out = {key: val if isinstance(val, Tensor) else val[0] for key, val in params.items()}
            obs_out = obs.val

        return params_out, obs_out


@attr.s(eq=False, auto_attribs=True)
class SetSBIHead(SetSBIMixin, SBIHead[_HeadOoutT, _KT], Generic[_HeadOoutT, _KT]):
    head: Union[Module, Callable[[SetBatch], _HeadOoutT]] = _empty_module

    def __attrs_post_init__(self):
        self.whitener = LazyWhitenOnline() if self.whiten else _empty_module

    def forward(self, params: _SBIParamsT, obs: Mapping[_KT, Sequence[Tensor]]):
        return params, self.head(SetBatch(self.whitener(_obs := self.prepare_obs({
            key: self._nested_cat(obs[key])
            for key in self.obs_names
        })), cast(LongTensor, _obs.new_tensor(
            one(unique_everseen(tuple(_.shape[self.set_dim] for _ in v) for v in obs.values())),
            dtype=int
        ))))


@attr.s(eq=False, auto_attribs=True)
class SetNRETail(ParamPackerNRETail[SetBatch, _KT], Generic[_KT], ABC):
    head: Union[BatchedSetModule, Callable[[SetBatch], Tensor]]
    tail: NRETail

    @abstractmethod
    def broadcast(self, theta: Tensor, obs: SetBatch) -> SetBatch: ...

    def _forward(self, theta: Tensor, obs: SetBatch, **kwargs) -> Tensor:
        return self.tail._forward(theta, self.head(self.broadcast(theta, obs)))


@attr.s(eq=False, auto_attribs=True)
class ConditionedSetNRETail(SetNRETail[_KT], Generic[_KT]):
    set_norm: Union[bool, SetNorm, Callable[[SetBatch, Tensor], SetBatch]] = True

    def __attrs_post_init__(self):
        if self.set_norm is True:
            self.set_norm = SetNorm()

    def broadcast(self, theta: Optional[Tensor], obs: SetBatch) -> SetBatch:
        return obs if theta is None else self.set_norm(obs, theta) if self.set_norm else obs.cat(theta)


@attr.s(eq=False, auto_attribs=True)
class CrossAttentionNRETail(SetNRETail[_KT], Generic[_KT]):
    encoder: CrossEncoder

    def broadcast(self, theta: Tensor, obs: SetBatch) -> SetBatch:
        return SetBatch.from_nested(self.encoder(theta, obs.expand_like(theta).to_nested()))


# class ARSetNRETail(ARNRETailComponent, ConditionedSetNRETail):
#     def pack_cond(self, obs: SetBatch):
#         return self.pack(OrderedDict((key, obs[key]) for key in self.cond_names))
#
#     def broadcast(self, theta: Tensor, obs: SetBatch):
#         return super().broadcast(
#             broadcast_cat((theta, self.pack_cond(obs)), -1)
#             if self.cond_names else theta, obs
#         )


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class LocalSetNRETail(SubsetSBIMixin, NRETail):
    summarize: bool = False
    shead: Union[BatchedSetModule, Callable[[SetBatch], Tensor]] = None

    def forward(self, params: _SBIParamsT, obs: SetBatch, **kwargs):
        params_out, obs_out = self._subsample({key: params[key] for key in self.param_names}, obs)

        if self.summarize:
            sobs = self.shead(obs)
            obs_out = broadcast_cat((obs_out, sobs.unsqueeze(self.set_dim+1)), -1)

        return super().forward(params_out, obs_out, **kwargs)


@attr.s(eq=False, auto_attribs=True, kw_only=True)
class SetCountsNRETail(SetSBIMixin, NRETail):
    whiten_counts: bool = True
    dtype: torch.dtype = None

    def __attrs_post_init__(self):
        self.counts_whitener = WhitenOnline((1,)) if self.whiten_counts else _empty_module

    def forward(self, params: _SBIParamsT, obs: SetBatch, **kwargs):
        return super().forward(params, self.counts_whitener(obs.sizes.to(
            self.dtype or torch.get_default_dtype()).unsqueeze(-1)))
