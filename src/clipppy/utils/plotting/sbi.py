from collections import ChainMap
from copy import copy
from dataclasses import dataclass
from functools import cached_property, cache
from itertools import combinations
from numbers import Number
from typing import Mapping, Any, Sequence, Literal, Union, cast, Iterable, MutableMapping

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from frozendict import frozendict
from matplotlib import pyplot as plt
from more_itertools import always_iterable
from torch import Tensor
from typing_extensions import TypeAlias, Self
from xarray import DataArray, Dataset
from xarray.core.weighted import DatasetWeighted

from . import to_percentiles
from ...sbi._typing import _KT, _MultiKT, MultiSBIProtocol

_CredT: TypeAlias = float

_ColorT: TypeAlias = Any
_MarkerT: TypeAlias = str


class MultiSBIPlotter:
    def __init__(
        self,
        samples: Mapping[_KT, Tensor],
        truths: Mapping[_KT, Tensor] = None, labels: Mapping[_KT, str] = frozendict(),
        sample_dims: tuple[_KT, ...] = ('sample',), dims: tuple[_KT, ...] = ('i',)
    ):
        self.truths = truths
        self.labels = labels
        self.sample_dims = sample_dims
        self.dims = sample_dims + dims

        self._samples = samples
        self.samples = Dataset({
            key: (self.dims[:val.ndim], val.detach().cpu().numpy())
            for key, val in samples.items()
        })

        self._log_weights: MutableMapping[_MultiKT, Tensor] = {}
        self._weights: MutableMapping[_MultiKT, DataArray] = {}
        self.cweights: MutableMapping[_MultiKT, DataArray] = {}

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value: Mapping[_MultiKT, Tensor]):
        for key, val in value.items():
            self._log_weights[key] = (val := val + self._log_weights.get(key, 0)).sub_(val.logsumexp(tuple(range(len(self.sample_dims)))))
            self._weights[key] = DataArray(weight := val.exp(), dims=self.dims[:val.ndim])
            self.cweights[key] = DataArray(to_percentiles(weight, len(self.sample_dims)), dims=self.dims[:val.ndim])

    def with_weights(self, log_weights: Mapping[_MultiKT, Tensor]):
        ret = copy(self)
        ret._log_weights = copy(self._log_weights)
        ret._weights = copy(self._weights)
        ret.cweights = copy(self.cweights)

        ret.weights = {key: r.detach().cpu() for key, r in log_weights.items()}
        return ret

    def eval(self, groups: Iterable[_MultiKT], net: MultiSBIProtocol, data: Mapping[str, Tensor]) -> Self:
        with torch.no_grad():
            headout = net.head(self._samples, {key: data[key] for key in net.obs_names})
            return self.with_weights({
                group: net.tail.forward_one(group, *headout)
                for group in groups
            })

    def _param_label(self, param_name: _KT) -> str:
        return self.labels.get(param_name, param_name)

    def _group_label(self, group: _MultiKT) -> str:
        return ', '.join(map(self._param_label, always_iterable(group)))

    @cached_property
    def ranges(self) -> Mapping[str, tuple[float, float]]:
        return {key: (val.min(self.sample_dims), val.max(self.sample_dims)) for key, val in self.samples.data_vars.items()}

    @cached_property
    def bin_edges(self) -> Mapping[str, np.ndarray]:
        return {
            key: np.histogram_bin_edges(val, 'auto')
            for key, val in self.samples.data_vars.items()
        }

    def mask_from_postmass(self, group: _MultiKT, thresh=1e-4):
        return self.cweights[group] < 1-thresh

    def mask_from_like_ratio(self, group: _MultiKT, thresh=1e-4):
        return self.weights[group] / self.weights[group].max(self.sample_dims) > thresh

    def _bounds(self, group: _MultiKT, thresh=1e-4, method: Literal['postmass', 'like_ratio'] = 'postmass') -> Mapping[str, tuple[Union[np.ndarray, Number], [np.ndarray, Number]]]:
        masked_samples = self.samples[list(always_iterable(group))].where((self.mask_from_postmass if method == 'postmass' else self.mask_from_like_ratio)(group, thresh))
        return {
            key: tuple(val.to_numpy())
            for key, val in pd.concat({
                'min': masked_samples.min(self.sample_dims).to_pandas().T,
                'max': masked_samples.max(self.sample_dims).to_pandas().T
            }, axis='index').groupby(level=1)
        }

    def bounds(self, group: _MultiKT, thresh=1e-4, method: Literal['postmass', 'like_ratio'] = 'postmass', **kwargs) -> Mapping[str, tuple[Tensor, Tensor]]:
        return {
            key: tuple(torch.tensor(v, **kwargs) for v in val)
            for key, val in self._bounds(group, thresh, method).items()
        }

    prior_kwargs = dict(label='prior')
    prior1d_kwargs = dict()
    prior2d_kwargs = dict()
    post_kwargs = dict(label='posterior')
    post1d_kwargs = dict()
    post2d_kwargs = dict()

    bounds_kwargs = dict(color='0.8', zorder=-1)

    def corner(
        self, group: _MultiKT,
        plot_prior=True, plot_hist1d=True, plot_hist2d=True, plot_kde1d=True, plot_kde2d=True, plot_bounds=True,
        prior_color: _ColorT = 'C1', prior_kwargs=frozendict(), prior1d_kwargs=frozendict(), prior2d_kwargs=frozendict(),
        post_color: _ColorT = 'C0', post_kwargs=frozendict(), post1d_kwargs=frozendict(), post2d_kwargs=frozendict(),
        levels: Sequence[float] = (0.68, 0.95, 1-1e-4), cut=True,
        truth2d_type: Literal['lines', 'marker'] = 'lines',
        truth_marker: _MarkerT = '*',
        truth_color: _ColorT = 'green',
        bounds_method_kwargs=frozendict(),
        bounds_kwargs=frozendict(),
        figsize=None
    ) -> tuple[plt.Figure, Union[np.ndarray, Sequence[Sequence[plt.Axes]]]]:
        if plot_bounds:
            bounds = self._bounds(group, **bounds_method_kwargs)

        ratio = self.weights[group]
        group = tuple(always_iterable(group))

        prior_kwargs = ChainMap(self.prior_kwargs, prior_kwargs)
        prior1d_kwargs = ChainMap(prior_kwargs, self.prior1d_kwargs, dict(color=prior_color), prior1d_kwargs)
        prior2d_kwargs = ChainMap(prior_kwargs, self.prior2d_kwargs, dict(cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list('cmap_prior', ((1, 1, 1, 0), prior_color))), prior2d_kwargs)

        post_kwargs = ChainMap(self.post_kwargs, post_kwargs)
        post1d_kwargs = ChainMap(post_kwargs, self.post1d_kwargs, dict(color=post_color), post1d_kwargs)
        post2d_kwargs = ChainMap(post_kwargs, self.post2d_kwargs, dict(cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list('cmap_post', ((1, 1, 1, 0), post_color))), post2d_kwargs)

        bounds_kwargs = ChainMap(self.bounds_kwargs, bounds_kwargs)

        if figsize is None:
            figsize = 3 * np.array(2 * (len(group),)) + 1

        fig, axs = plt.subplots(len(group), len(group), figsize=figsize, squeeze=False)

        axs = np.atleast_2d(axs)

        for ax in axs.flatten():
            ax.set_axis_off()
        for ax in axs[:-1].flatten():
            ax.set_xticklabels([])
        for ax in axs[:, 1:].flatten():
            ax.set_yticklabels([])

        for i, param in enumerate(group):
            axs[i, i].set_xlabel(None)  # because seaborn sets it automatically
            if i:
                axs[i, 0].set_ylabel(self._param_label(param))
            axs[-1, i].set_xlabel(self._param_label(param))

            ax: plt.Axes = axs[i, i]
            ax.set_axis_on()
            ax.set_yticklabels([])

            if (truth := self.truths.get(param, None)) is not None:
                ax.axvline(truth, color=truth_color, label='truth')

            kwargs = dict(x=self.samples[param], ax=ax)

            if plot_hist1d:
                histkwargs = dict(**kwargs, stat='density', bins=self.bin_edges[param], kde=plot_kde1d)

                if plot_prior:
                    sns.histplot(**histkwargs, **prior1d_kwargs)
                sns.histplot(**histkwargs, **post1d_kwargs, weights=ratio)

            elif plot_kde1d:
                kdekwargs = dict(**kwargs, cut=cut, ax=ax)
                if plot_prior:
                    sns.kdeplot(**kdekwargs, **prior1d_kwargs)
                sns.kdeplot(**kdekwargs, **post1d_kwargs, weights=ratio)

            if plot_bounds:
                ax.axvspan(self.ranges[param][0], bounds[param][0], **bounds_kwargs)
                ax.axvspan(bounds[param][1], self.ranges[param][1], **bounds_kwargs)

            ax.set_xlim(*self.ranges[param])
            ax.set_ylim(bottom=0)
            ax.set_ylabel('')

            ax.legend()

        for (i1, param1), (i2, param2) in combinations(enumerate(group), 2):
            ax: plt.Axes = axs[i2, i1]
            ax.set_axis_on()

            kwargs = dict(x=self.samples[param1], y=self.samples[param2])

            if plot_hist2d:
                histkwargs = dict(**kwargs, bins=(self.bin_edges[param1], self.bin_edges[param2]), density=True)
                hist_prior, *_ = np.histogram2d(**histkwargs)
                hist_post, xedges, yedges = np.histogram2d(**histkwargs, weights=ratio)
                imkwargs = dict(extent=(*xedges[(0, -1),], *yedges[(0, -1),]), origin='lower')

                if plot_prior:
                    ax.imshow(hist_prior.T, **imkwargs, **{**dict(alpha=(hist_prior.max()/hist_post.max()).clip(0, 1)), **prior2d_kwargs})
                ax.imshow(hist_post.T, **imkwargs, **post2d_kwargs)

            if plot_kde2d:
                kdekwargs = dict(**kwargs, cut=cut, levels=[1-l for l in sorted(levels, reverse=True)], ax=ax)
                if plot_prior:
                    sns.kdeplot(**kdekwargs, **{**dict(color=prior_color), **prior_kwargs})
                sns.kdeplot(**kdekwargs, **{**dict(color=post_color), **post_kwargs}, weights=ratio)

            if plot_bounds:
                ax.axvspan(self.ranges[param1][0], bounds[param1][0], **bounds_kwargs)
                ax.axvspan(bounds[param1][1], self.ranges[param1][1], **bounds_kwargs)
                ax.axhspan(self.ranges[param2][0], bounds[param2][0], **bounds_kwargs)
                ax.axhspan(bounds[param2][1], self.ranges[param2][1], **bounds_kwargs)

            ax.set_xlim(*self.ranges[param1])
            ax.set_ylim(*self.ranges[param2])
            ax.set_aspect('auto')

            truth_x, truth_y = (self.truths.get(key, None) for key in (param1, param2))

            if truth2d_type == 'lines':
                if truth_x is not None:
                    ax.axvline(truth_x, color=truth_color)
                if truth_y is not None:
                    ax.axhline(truth_y, color=truth_color)
            if truth2d_type == 'marker' and None not in (truth_x, truth_y):
                ax.plot(self.truths[param1], self.truths[param2], truth_marker, color=truth_color)

        return fig, axs

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

    @cache
    def _stats(self, key: _MultiKT, cred: _CredT) -> Stats:
        return self.Stats(self._weighted(key), self.sample_dims, cred)

    def _latent_v_truth(self, param_name: _KT, cred: float, ax: plt.Axes, **kwargs) -> plt.Axes:
        s = self._stats(param_name, cred)

        ax.errorbar(
            self.truths[param_name], s.means[param_name],
            ((s.meds-s.los)[param_name], (s.his-s.meds)[param_name]),
            **{**dict(
                ls='none', marker='.', markersize=5, markeredgecolor='none',
                elinewidth=0.5, capsize=2, capthick=0.5
            ), **kwargs}
        )

        return ax

    def latent_v_truth(self, param_name: _KT, cred: float, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        if ax is None:
            ax = plt.gca()

            ax.plot(*2*((self.truths[param_name].min(), self.truths[param_name].max()),), 'k--')
            ax.set_xlabel('true')
            ax.set_ylabel('inferred')
            ax.set_title(self._param_label(param_name))

        return self._latent_v_truth(param_name, cred, ax, **kwargs)
