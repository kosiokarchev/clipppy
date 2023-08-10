from copy import copy
from dataclasses import dataclass
from typing import Mapping, Iterable, cast

import torch
from importlib_metadata import always_iterable
from more_itertools import value_chain
from torch import Tensor
from typing_extensions import TypeAlias
from xarray import Dataset, DataArray
from xarray.core.weighted import DatasetWeighted

from ....sbi._typing import _KT, _MultiKT, MultiSBIProtocol

_CredT: TypeAlias = float


# class MultiLatentPlotter:
#     def __init__(
#         self,
#         samples: Mapping[_KT, Tensor], ratios: Mapping[_MultiKT, Tensor],
#         groups: Iterable[_MultiKT], truths: Mapping[_KT, Tensor] = None,
#         dims: tuple[_KT] = ('i',), sample_dims: tuple[_KT] = ('sample',)
#     ):
#         self.groups = groups
#         self.truths = truths
#         self.sample_dims = sample_dims
#
#         self.samples = Dataset({
#             key: (self.sample_dims + dims[:val.ndim-1], val)
#             for key in value_chain(*groups)
#             for val in [samples[key].detach().cpu().numpy()]
#         })
#         self.weights: Mapping[_MultiKT, DataArray] = {
#             key: DataArray(r / r.sum(0), dims=self.sample_dims + dims[:r.ndim-1])
#             for key, r in ratios.items() for r in [r.detach().cpu().numpy()]
#         }
#
#     @dataclass
#     class Stats:
#         d: DatasetWeighted
#         sample_dims: tuple[_KT]
#
#         cred: _CredT = None
#
#         @property
#         def means(self):
#             return self.d.mean(dim=self.sample_dims)
#
#         @property
#         def stds(self):
#             return self.d.std(dim=self.sample_dims)
#
#         @property
#         def meds(self):
#             return self.d.quantile(0.5, dim=self.sample_dims)
#
#         @property
#         def los(self):
#             return self.d.quantile((1-self.cred)/2, dim=self.sample_dims)
#
#         @property
#         def his(self):
#             return self.d.quantile((1+self.cred)/2, dim=self.sample_dims)
#
#
#     def _weighted(self, key: _MultiKT):
#         return self.samples[list(always_iterable(key))].weighted(self.weights[key])
#
#     def _stats(self, key: _MultiKT, cred: _CredT) -> Stats:
#         return self.Stats(self._weighted(key), self.sample_dims, cred)
#
#     def _plot_1d(self, param_name, cred, ax, **kwargs):
#         s = self._stats(param_name, cred)
#
#         ax.errorbar(
#             self.truths[param_name], s.means[param_name],
#             ((s.meds-s.los)[param_name], (s.his-s.meds)[param_name]),
#             **{**dict(
#                 ls='none', marker='none', markersize=5, markeredgecolor='none',
#                 elinewidth=0.5, capsize=2, capthick=0.5
#             ), **kwargs}
#         )
#
#         return ax
#
#     def plot_1d(self, param_name: _KT, cred, ax=None, **kwargs):
#         if ax is None:
#             from matplotlib import pyplot as plt
#
#             ax = plt.gca()
#
#             ax.plot(*2*((self.truths[param_name].min(), self.truths[param_name].max()),), 'k--')
#             ax.set_xlabel('true')
#             ax.set_ylabel('inferred')
#             ax.set_title(param_name)
#
#         return self._plot_1d(param_name, cred, ax, **kwargs)
#
#     def bounds(self, thresh, *args, **kwargs) -> Mapping[_KT, tuple[Tensor, Tensor]]:
#         ret = [{}, {}]
#
#         for key in self.groups:
#             s = self._stats(key, 1-thresh)
#             for i, b in enumerate((s.los, s.his)):
#                 ret[i].update({
#                     k: torch.from_numpy(v.to_numpy()).to(*args, **kwargs)
#                     for k, v in b.items()
#                 })
#         return {
#             key: (ret[0][key], ret[1][key])
#             for key in value_chain(*self.groups)
#         }

class MultiLatentPlotter:
    weights: Mapping[_MultiKT, DataArray] = None

    def __init__(
        self,
        samples: Mapping[_KT, Tensor],
        truths: Mapping[_KT, Tensor] = None,
        sample_dims: tuple[_KT] = ('sample',), dims: tuple[_KT] = ('i',)
    ):
        self.truths = truths
        self.sample_dims = sample_dims
        self.dims = sample_dims + dims

        self._samples = samples
        self.samples = Dataset({
            key: (self.dims[:val.ndim], val.detach().cpu().numpy())
            for key, val in samples.items()
        })

    def eval(self, groups: Iterable[_MultiKT], nre: MultiSBIProtocol, trace: Mapping[str, Tensor]):
        with torch.no_grad():
            headout = nre.head(self._samples, {key: trace[key] for key in nre.obs_names})
            return self.with_weights({
                group: nre.tail.forward_one(group, *headout)
                for group in groups
            })

    def with_weights(self, log_weights: Mapping[_MultiKT, Tensor]):
        ret = copy(self)
        ret.weights = {
            key: DataArray((r - r.logsumep(-1)).exp(), dims=self.dims[:r.ndim])
            for key, r in log_weights.items()
            for r in [r.detach().cpu().numpy()]
        }
        return ret

    @dataclass
    class Stats:
        d: DatasetWeighted
        sample_dims: tuple[_KT]

        cred: _CredT = None

        @property
        def means(self):
            return self.d.mean(dim=self.sample_dims)

        @property
        def stds(self):
            return self.d.std(dim=self.sample_dims)

        @property
        def meds(self):
            return self.d.quantile(0.5, dim=self.sample_dims)

        @property
        def los(self):
            return self.d.quantile((1-self.cred)/2, dim=self.sample_dims)

        @property
        def his(self):
            return self.d.quantile((1+self.cred)/2, dim=self.sample_dims)


    def _weighted(self, key: _MultiKT) -> DatasetWeighted:
        return cast(Dataset, self.samples[list(always_iterable(key))]).weighted(self.weights[key])

    def _stats(self, key: _MultiKT, cred: _CredT) -> Stats:
        return self.Stats(self._weighted(key), self.sample_dims, cred)

    def _plot_1d(self, param_name, cred, ax, **kwargs):
        s = self._stats(param_name, cred)

        ax.errorbar(
            self.truths[param_name], s.means[param_name],
            ((s.meds-s.los)[param_name], (s.his-s.meds)[param_name]),
            **{**dict(
                ls='none', marker='none', markersize=5, markeredgecolor='none',
                elinewidth=0.5, capsize=2, capthick=0.5
            ), **kwargs}
        )

        return ax

    def plot_1d(self, param_name: _KT, cred, ax=None, **kwargs):
        if ax is None:
            from matplotlib import pyplot as plt

            ax = plt.gca()

            ax.plot(*2*((self.truths[param_name].min(), self.truths[param_name].max()),), 'k--')
            ax.set_xlabel('true')
            ax.set_ylabel('inferred')
            ax.set_title(param_name)

        return self._plot_1d(param_name, cred, ax, **kwargs)

    def bounds(self, thresh, *args, **kwargs) -> Mapping[_KT, tuple[Tensor, Tensor]]:
        ret = [{}, {}]

        for key in self.weights.keys():
            s = self._stats(key, 1-thresh)
            for i, b in enumerate((s.los, s.his)):
                ret[i].update({
                    k: torch.from_numpy(v.to_numpy()).to(*args, **kwargs)
                    for k, v in b.items()
                })
        return {
            key: (ret[0][key], ret[1][key])
            for key in value_chain(*self.weights.keys())
        }
