from functools import reduce, partial
from itertools import chain
from operator import add
from typing import Iterable, Mapping, Generic, Iterator, Optional, Callable, Union

import attr
from more_itertools import always_iterable, collapse
from torch import Tensor
from typing_extensions import TypeAlias

from ..commands.lightning.loss import BaseSBILoss
from ..commands.lightning.nre import NRE
from ..sbi._typing import _MultiKT, _SBIObsT, _SBIParamsT, _KT
from ..sbi.multi import PackerMixin
from ..sbi.nn import BaseMultiSBITail, MultiSBITail
from ..sbi.nn.nre import BaseNRETail


@attr.s
class ARNRETailComponent(Generic[_KT]):
    cond_names: Iterable[_KT] = attr.ib(default=None, kw_only=True)


_PostTailT: TypeAlias = BaseNRETail[_SBIObsT, _KT]
_PriorTailT: TypeAlias = BaseNRETail[_SBIParamsT, _KT]
_CondTailT: TypeAlias = Callable[[_SBIParamsT], Tensor]


@attr.s
class ARNRETail(BaseMultiSBITail[_SBIObsT, Tensor, _KT], Generic[_KT]):
    tails: Mapping[_MultiKT, _PostTailT]
    prior_tails: Mapping[_MultiKT, _PriorTailT] = attr.ib(factory=dict)

    def cond_tail(self, key: _MultiKT, params: _SBIParamsT, obs: _SBIObsT, **kwargs) -> tuple[_CondTailT, Optional[_CondTailT]]:
        return (
            partial(self.tails[key], obs=dict(chain(
                (conds := {k: params[k] for k in self.cond_names_map[key]}).items(),
                obs.items())), **kwargs),
            partial(self.prior_tails[key], obs=conds, **kwargs)
            if key in self.prior_tails else None
        )

    def cond_tails(self, params: _SBIParamsT, obs: _SBIObsT, **kwargs) -> Iterator[tuple[_MultiKT, _CondTailT, Optional[_CondTailT]]]:
        for key in self.tails.keys():
            yield key, *self.cond_tail(key, params, obs, **kwargs)

    cond_names_map: Mapping[_MultiKT, Iterable[_KT]] = attr.ib()
    @cond_names_map.default
    def _(self):
        return {
            key: cond for newcond in [()]
            for key in self.tails.keys()
            for cond in [newcond]
            for newcond in [(*newcond, *always_iterable(key))]
        }

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self._add_tails(self.prior_tails, 'prior_')

        self.param_names = tuple(collapse(self.tails.keys()))

        for key, tail in self.tails.items():
            if isinstance(tail, ARNRETailComponent) and tail.cond_names is None:
                tail.cond_names = self.cond_names_map[key]
        for key, tail in self.prior_tails.items():
            if isinstance(tail, PackerMixin):
                if tail.param_names is None:
                    tail.param_names = *always_iterable(key),
                if tail.obs_names is None:
                    tail.obs_names = self.cond_names_map[key]

    def forward_one(self, key: _KT, params: _SBIParamsT, obs: _SBIObsT, **kwargs) -> Tensor:
        tail, prior_tail = self.cond_tail(key, params, obs, **kwargs)
        return tail(params) if prior_tail is None else (
            tail(params) - prior_tail(params).clamp(min=0)
        )

    def forward(self, params: _SBIParamsT, obs: _SBIObsT, **kwargs) -> Mapping[Iterable[_KT], Tensor]:
        return reduce(add, super().forward(params, obs, **kwargs).values())


class ARNRE(NRE[_SBIObsT, _KT], Generic[_KT]):
    tail: Union[ARNRETail[_KT], MultiSBITail[_SBIObsT, Tensor, _KT]]

    @staticmethod
    def _iter_cond_artails(artail: ARNRETail, obs, params, **kwargs) -> Iterator[tuple[_MultiKT, _CondTailT]]:
        for key, tail, prior_tail in artail.cond_tails(params, obs, **kwargs):
            yield key, tail
            if prior_tail is not None:
                yield ('prior', key), prior_tail

    def _iter_cond_tails(self, obs, params, **kwargs) -> Iterator[tuple[_MultiKT, _CondTailT]]:
        if isinstance(self.tail, ARNRETail):
            yield from self._iter_cond_artails(self.tail, obs, params, **kwargs)
        elif isinstance(self.tail, MultiSBITail):
            for key, tail in self.tail.tails.items():
                if isinstance(tail, ARNRETail):
                    yield from self._iter_cond_artails(tail, obs, params, **kwargs)
                else:
                    yield key, partial(tail, obs=obs, **kwargs)
        else:
            raise TypeError

    def _loss_one(self, obs, params, decoys, **kwargs) -> BaseSBILoss.ReturnT:
        return self.lossfunc(*map(dict, zip(*(
            ((key, tail(params)), (key, tail(decoys)))
            for key, tail in self._iter_cond_tails(obs, params)
        ))))
