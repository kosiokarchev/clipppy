from abc import ABC
from collections import OrderedDict
from typing import Callable, Union, Mapping, Sequence, cast, Iterable, Generic, Optional

import attr
import torch
from more_itertools import one
from more_itertools import unique_everseen
from torch import Tensor, LongTensor

from .arnre import ARNRETailComponent
from ..sbi._typing import _SBIObsT, _KT
from ..sbi.multi import dict_to_vect
from ..sbi.nn import BaseSBITail, _TailOutT
from ..sbi.nn.nre import NRETail, ParamPackerNRETail
from ..utils.nn import LazyWhitenOnline
from ..utils.nn.empty import _empty_module
from ..utils.nn.sets import BatchedSetModule

from phytorchx import broadcast_cat


@attr.s(auto_attribs=True, eq=False)
class SetSBITail(BaseSBITail[_SBIObsT, _TailOutT, _KT], Generic[_TailOutT, _KT], ABC):
    set_dim: int = attr.ib(default=0, kw_only=True)
    event_dims: Mapping[_KT, int] = attr.ib(factory=dict, kw_only=True)

    def prepare_obs(self, obs: _SBIObsT) -> Tensor:
        return dict_to_vect(obs, self.event_dims)

    def _nested_cat(self, nt: Sequence[Tensor]):
        return torch.cat(tuple(t.movedim(self.set_dim, 0) for t in nt), 0)



@attr.s(auto_attribs=True, eq=False)
class SetNRETail(SetSBITail[Tensor, _KT], ParamPackerNRETail[Tensor, _KT], Generic[_KT]):
    head: Union[BatchedSetModule, Callable[[Tensor, LongTensor], Tensor]]
    tail: NRETail

    obs_names: Iterable[_KT]

    whiten: bool = True

    def __attrs_post_init__(self):
        self.whitener = LazyWhitenOnline() if self.whiten else _empty_module

    def broadcast(self, theta: Optional[Tensor], obs: _SBIObsT):
        x = self.prepare_obs(OrderedDict((key, self._nested_cat(obs[key])) for key in self.obs_names))
        sizes = cast(LongTensor, x.new_tensor(
            one(unique_everseen(tuple(_.shape[self.set_dim] for _ in obs[key]) for key in self.obs_names)),
            dtype=torch.int
        ))

        if theta is None:
            return x, sizes
        else:
            if len(sizes) == 1 and len(theta.shape) > 1:
                sizes = sizes.repeat(theta.shape[-2])
                x = x.expand(theta.shape[-2], *x.shape).flatten(end_dim=1)

            return broadcast_cat((
                theta.expand(*theta.shape[:-2], len(sizes), theta.shape[-1]).repeat_interleave(sizes, -2),
                x
            ), dim=-1), sizes

    def _forward(self, theta: Tensor, obs: _SBIObsT, **kwargs) -> Tensor:
        x, sizes = self.broadcast(theta, obs)
        return self.tail._forward(theta, self.head(self.whitener(x), sizes), **kwargs)


@attr.s(eq=False)
class AbstractARSetNRETail(ARNRETailComponent, SetNRETail):
    def pack_cond(self, obs: _SBIObsT):
        return self.pack(OrderedDict((key, obs[key]) for key in self.cond_names))


@attr.s(eq=False)
class ARSetNRETail(AbstractARSetNRETail):
    def broadcast(self, theta: Tensor, obs: _SBIObsT):
        return super().broadcast(
            broadcast_cat((theta, self.pack_cond(obs)), -1)
            if self.cond_names else theta, obs
        )


@attr.s(eq=False)
class LocalARSetNRETail(AbstractARSetNRETail):
    def broadcast(self, theta: Tensor, obs: _SBIObsT):
        return super().broadcast(
            self.pack_cond(obs) if self.cond_names else None, obs
        )
