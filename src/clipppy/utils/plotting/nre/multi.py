from abc import abstractmethod
from functools import partialmethod, cached_property, reduce
from operator import or_
from typing import Iterable, Callable, Sequence, Type, Mapping, Literal

import attr
import torch
from matplotlib import pyplot as plt
from more_itertools import always_iterable, last
from torch import Tensor
from torch.nn import Module
from typing_extensions import TypeAlias

from . import BaseNREPlotter, _MultiTailKT, NREPlotter, _HeadT, _qT
from ....sbi._typing import MultiSBIProtocol, _SBIObsT
from ....sbi.nn import MultiSBITail


_MultiRatioT: TypeAlias = Mapping[_MultiTailKT, Tensor]


class PseudoTail(Module):
    def __init__(self, key, tail: MultiSBITail):
        super().__init__()
        self.key = key
        self.tail = tail

    def forward(self, *args, **kwargs):
        return self.tail.forward_one(self.key, *args, **kwargs)


class MappedMixin:
    @abstractmethod
    def _mapped(self, funcname, *args, **kwargs): ...

    def __init_subclass__(cls, *, mapped_funcs: Iterable[Callable] = (), **kwargs):
        super().__init_subclass__(**kwargs)
        for func in mapped_funcs:
            setattr(cls, func.__name__, partialmethod(cls._mapped, func.__name__))


@attr.s(auto_attribs=True)
class MultiNREPlotter(MappedMixin, BaseNREPlotter, mapped_funcs=(
        BaseNREPlotter.log_ratio, BaseNREPlotter.ratio)):
    groups: Sequence[_MultiTailKT]
    subcls: Type[BaseNREPlotter] = NREPlotter

    @cached_property
    def plotters(self) -> Mapping[_MultiTailKT, NREPlotter]:
        return {g: self.subcls(
            params=self.params,
            param_names=tuple(always_iterable(g)),
            labels=self.labels
        ) for g in self.groups}

    def subtails(self, tail: MultiSBITail):
        return {g: PseudoTail(g, tail) for g in self.groups}

    def _mapped(self, funcname, obs: _SBIObsT, head: _HeadT, tail: MultiSBITail, *args, **kwargs) -> Mapping[_MultiTailKT, Tensor]:
        return {g: getattr(self.plotters[g], funcname)(obs, head, subtail, *args, **kwargs) for g, subtail in self.subtails(tail).items()}

    def _perc(self, log_ratios: _MultiRatioT, log_ratio: _MultiRatioT):
        return {g: self.plotters[g]._perc(log_ratios[g], log_ratio[g]) for g in self.groups}

    def corner(self, ratio: _MultiRatioT, **kwargs):
        return {g: self.plotters[g].corner(r, **kwargs) for g, r in ratio.items()}

    def bounds(self, ratio: _MultiRatioT, thresh=1e-4, method: Literal['postmass', 'like_ratio'] = 'postmass') -> Mapping[str, tuple[float, float]]:
        return reduce(or_, (self.plotters[group].bounds(r, thresh, method) for group, r in ratio.items()))

    def qq(self, qs: Mapping[str, _qT] = (), ax: plt.Axes = None, **kwargs):
        ax = last(
            ax for ax in [ax]
            for g, val in qs.items()
            for ax in [self.plotters[g].qq(val, label=self.group_label(g), ax=ax, **kwargs)]
        )
        ax.legend()
        return ax


def multi_posterior(nre: MultiSBIProtocol, nrep: MultiNREPlotter, trace, **kwargs) -> Mapping[_MultiTailKT, plt.Figure]:
    return {g: ret[0] for g, ret in nrep.corner(nrep.ratio({
        key: val.unsqueeze(-nre.head.event_dims.get(key, val.ndim)-1)
        for key in nre.obs_names for val in [trace[key]]
    }, nre.head, nre.tail), truths={
        key: float(trace[key]) for key in nre.param_names
        if not torch.is_tensor(trace[key]) or trace[key].numel() == 1
    }, **kwargs).items()}
