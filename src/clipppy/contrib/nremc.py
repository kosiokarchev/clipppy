from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Generic, TypeVar

import numpy as np
import pyro
import torch
from frozendict import frozendict
from more_itertools import always_iterable
from pyro.nn import PyroModule
from scipy.stats import gaussian_kde
from torch import Tensor, BoolTensor

from ..distributions.infinite import InfiniteUniform
from ..sbi._typing import _MultiKT, _SBIParamsT
from ..sbi.multi import PackerMixin

_ParamsT = TypeVar('_ParamsT')
_OutT = TypeVar('_OutT')


class Bound(Generic[_ParamsT, _OutT]):
    def contains(self, params: _ParamsT) -> _OutT: ...


@dataclass
class DynestyBound(Bound[_SBIParamsT, Tensor | BoolTensor]):
    bound: Bound[np.ndarray, bool]
    param_names: Iterable[str]

    def contains(self, params: _SBIParamsT) -> Tensor | BoolTensor:
        return torch.tensor([
            self.bound.contains(x)
            for x in np.atleast_2d(np.stack([
                params[key].numpy(force=True)
                for key in self.param_names
            ], -1))
        ])


class KPE(PackerMixin):
    def __init__(self, groups: Iterable[_MultiKT], X: _SBIParamsT, kde_kwargs=frozendict(), **kwargs):
        super().__init__(**kwargs)
        self.kdes = {
            group: gaussian_kde(self.packed(group, X).T)
            # group: cast(KernelDensity, KernelDensity(**{**dict(bandwidth='scott'), **kde_kwargs}).fit(self.packed(group, X)))
            for group in groups
        }

    def packed(self, group: _MultiKT, val: _SBIParamsT):
        return self.pack({k: val[k] for k in always_iterable(group)})

    def log_prob(self, x: _SBIParamsT) -> Mapping[_MultiKT, Tensor]:
        return {group: torch.tensor(
            # kde.score_samples(torch.atleast_2d(self.packed(group, x)))
            kde.logpdf(torch.atleast_2d(self.packed(group, x)).T)
        ) for group, kde in self.kdes.items()}


@dataclass
class BoundedNREProb:
    param_names: Iterable[str]

    log_ratio: Callable[[_SBIParamsT], Mapping[_MultiKT, Tensor]] = field(default=None, kw_only=True)
    bound: Bound[_SBIParamsT, BoolTensor] = field(default=None, kw_only=True)

    def get_params(self):
        return {
            key: pyro.sample(key, InfiniteUniform())
            for key in self.param_names
        }

    def _ratio_factor(self, params):
        return self.log_ratio(params) if self.log_ratio else {}

    def __call__(self, *args, **kwargs):
        params = self.get_params()

        for key, val in self._ratio_factor(params).items():
            pyro.factor('_ratio'+str(key), val)

        if self.bound is not None:
            pyro.factor('_bound', torch.where(
                self.bound.contains(params),
                0, -float('inf')
            ))


@dataclass
class ModelNREProb(BoundedNREProb):
    model: PyroModule

    def get_params(self):
        with self.model._pyro_context:
            return {key: getattr(self.model, key) for key in self.param_names}


# @dataclass
# class NREMC:
#     model: PyroModule
#     param_names: Iterable[str]
#     log_ratio: Callable[[_SBIParamsT], Mapping[_MultiKT, Tensor]]
#
#     def _ratio_factor(self, params):
#         return self.log_ratio(params)
#
#     def __call__(self, *args,  **kwargs):
#         with self.model._pyro_context:
#             params = {key: getattr(self.model, key) for key in self.param_names}
#
#         for key, val in self._ratio_factor(params).items():
#             pyro.factor('_ratio'+str(key), val)
#
#
# @dataclass
# class TNREPrior(NREMC):
#     log_ratio_thresh: Number
#
#     def _ratio_factor(self, params):
#         return {
#             key: torch.where(val < self.log_ratio_thresh, -float('inf'), 0)
#             for key, val in super()._ratio_factor(params).items()
#         }
