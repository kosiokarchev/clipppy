from collections import ChainMap
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations, starmap
from numbers import Number
from typing import Mapping, Any, Sequence, Literal, Union, cast, Iterable, MutableMapping

import attr
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from frozendict import frozendict
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from more_itertools import always_iterable, consume
from scipy.stats import halfnorm
from torch import Tensor
from typing_extensions import TypeAlias, Self
from xarray import DataArray, Dataset
from xarray.core.weighted import DatasetWeighted

from . import to_percentiles
from ...sbi._typing import _KT, _MultiKT, _SBIObsT, BaseMultiSBIResultRep, _MultiMappingT, MultiNREProtocol

_CredT: TypeAlias = float

_ColorT: TypeAlias = Any
_MarkerT: TypeAlias = str


@attr.define(slots=False)
class MultiSBIPlotter(BaseMultiSBIResultRep):
    labels: Mapping[_KT, str] = attr.ib(factory=dict)

    def param_label(self, param_name: _KT) -> str:
        return self.labels.get(param_name, param_name)

    def group_label(self, group: _MultiKT) -> str:
        return ', '.join(map(self.param_label, always_iterable(group)))


@attr.define(slots=False)
class MultiSBIPosteriorPlotter(MultiSBIPlotter):
    truths: Mapping[_KT, Tensor] = attr.ib(factory=dict)
    sample_dims: tuple[_KT, ...] = ('sample',)
    local_dims: tuple[_KT, ...] = ('i',)

    dims: tuple[_KT, ...] = attr.field(init=False)
    dims.default(lambda self: self.sample_dims + self.local_dims)

    def __attrs_post_init__(self):
        self.samples = Dataset({
            key: (self.dims[:val.ndim], val.numpy(force=True))
            for key, val in self._samples.items()
        })

    _log_weights: MutableMapping[_MultiKT, Tensor] = attr.field(init=False, factory=dict)
    _weights: MutableMapping[_MultiKT, DataArray] = attr.field(init=False, factory=dict)
    cweights: MutableMapping[_MultiKT, DataArray] = attr.field(init=False, factory=dict)

    def copy(self, **kwargs):
        ret = copy(self)
        consume(starmap(ret.__setattr__, kwargs.items()))

        ret._log_weights = copy(self._log_weights)
        ret._weights = copy(self._weights)
        ret.cweights = copy(self.cweights)

        return ret

    # WEIGHTING

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, log_ratios: Mapping[_MultiKT, Tensor]):
        for key, val in log_ratios.items():
            self._log_weights[key] = (val := val + self._log_weights.get(key, 0)).sub_(val.logsumexp(tuple(range(len(self.sample_dims)))))
            self._weights[key] = DataArray(weight := val.exp(), dims=self.dims[:val.ndim])
            self.cweights[key] = DataArray(to_percentiles(weight, len(self.sample_dims)), dims=self.dims[:val.ndim])

    def with_ratios(self, log_ratios: Mapping[_MultiKT, Tensor], **kwargs):
        ret = self.copy(**kwargs)
        ret.weights = {key: r.detach().cpu() for key, r in log_ratios.items()}
        return ret

    def eval_nre(self, groups: Iterable[_MultiKT], net: MultiNREProtocol, data: _SBIObsT, **kwargs) -> Self:
        return self.with_ratios(self._eval_nre(groups, net, self._samples, {key: data[key] for key in net.obs_names}), **kwargs)

    # end WEIGHTING

    @cached_property
    def ranges(self) -> Mapping[str, tuple[float, float]]:
        return {key: (val.min(self.sample_dims), val.max(self.sample_dims)) for key, val in self.samples.data_vars.items()}

    @cached_property
    def bin_edges(self) -> Mapping[str, np.ndarray]:
        return {
            key: np.histogram_bin_edges(val, 'auto')
            for key, val in self.samples.data_vars.items()
        }

    # BOUNDS

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

    # end BOUNDS

    # CORNER

    prior_color: _ColorT = 'tab:orange'
    prior_kwargs: Mapping[str, Any] = attr.field(default={}, converter=dict(label='prior').__or__)
    prior1d_kwargs: Mapping[str, Any] = dict()
    prior2d_kwargs: Mapping[str, Any] = dict()

    post_color: _ColorT = 'tab:blue'
    post_kwargs: Mapping[str, Any] = attr.field(default={}, converter=dict(label='posterior').__or__)
    post1d_kwargs: Mapping[str, Any] = dict()
    post2d_kwargs: Mapping[str, Any] = dict()

    bounds_kwargs: Mapping[str, Any] = attr.field(default={}, converter=dict(color='0.8', zorder=-1).__or__)

    def _corner(self, group: Sequence[_KT], figsize=None) -> tuple[plt.Figure, Union[np.ndarray, Sequence[Sequence[plt.Axes]]]]:
        if figsize is None:
            figsize = 3 * np.array(2 * (len(group),)) + 1

        fig, axs = plt.subplots(len(group), len(group), sharex='col', sharey='row', figsize=figsize, squeeze=False)

        for ax in axs[np.triu_indices_from(axs, 1)]:
            ax.remove()
        for ax in np.diagonal(axs):
            # TODO: dependence of uplot
            from uplot.utils import unshare

            unshare(ax, 'y')
            ax.yaxis.set_visible(False)

        for ax, param in zip(axs[1:, 0], group[1:]):
            ax.set_ylabel(self.param_label(param))
            ax.set_ylim(*self.ranges[param])

        # for ax in axs[:-1].flatten():
        #     ax.set_xlabel()
        for ax, param in zip(axs[-1], group):
            ax.set_xlabel(self.param_label(param))
            ax.set_xlim(*self.ranges[param])

        return fig, axs

    def corner(
        self, group: _MultiKT, *,
        axs: Union[np.ndarray, Sequence[Sequence[plt.Axes]]] = None,
        plot_prior=True, plot_hist1d=True, plot_hist2d=True, plot_kde1d=True, plot_kde2d=True, plot_truth=True, plot_bounds=True,
        prior_color: _ColorT = None, prior_kwargs=frozendict(), prior1d_kwargs=frozendict(), prior2d_kwargs=frozendict(),
        post_color: _ColorT = None, post_kwargs=frozendict(), post1d_kwargs=frozendict(), post2d_kwargs=frozendict(),
        levels: Sequence[float] = (0.39346934, 0.86466472, 1-1e-4), cut=0,
        truth2d_type: Literal['lines', 'marker'] = 'lines',
        truth_marker: _MarkerT = '*',
        truth_color: _ColorT = 'tab:green',
        bounds_method_kwargs=frozendict(),
        bounds_kwargs=frozendict(),
        figsize=None
    ) -> tuple[plt.Figure, Union[np.ndarray, Sequence[Sequence[plt.Axes]]]]:
        if plot_bounds:
            bounds = self._bounds(group, **bounds_method_kwargs)

        ratio = self.weights[group]
        group = tuple(always_iterable(group))

        if prior_color is None:
            prior_color = self.prior_color
        prior_kwargs = ChainMap(prior_kwargs, self.prior_kwargs)
        prior_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('cmap_prior', ((1, 1, 1, 0), prior_color))
        prior1d_kwargs = ChainMap(prior1d_kwargs, self.prior1d_kwargs, prior_kwargs, dict(color=prior_color))
        prior2d_kwargs = ChainMap(prior2d_kwargs, self.prior2d_kwargs, prior_kwargs)

        if post_color is None:
            post_color = self.post_color
        post_kwargs = ChainMap(post_kwargs, self.post_kwargs)
        post_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('cmap_post', ((1, 1, 1, 0), post_color))
        post1d_kwargs = ChainMap(post1d_kwargs, self.post1d_kwargs, post_kwargs, dict(color=post_color))
        post2d_kwargs = ChainMap(post2d_kwargs, self.post2d_kwargs, post_kwargs)

        bounds_kwargs = ChainMap(bounds_kwargs, self.bounds_kwargs)

        if axs is None:
            fig, axs = self._corner(group, figsize)
        else:
            fig = axs[0][0].figure

        for i, param in enumerate(group):
            ax: plt.Axes = axs[i, i]

            if plot_truth and (truth := self.truths.get(param, None)) is not None:
                ax.axvline(truth, color=truth_color, label='truth')

            kwargs = dict(x=self.samples[param], ax=ax)

            if plot_hist1d:
                histkwargs = dict(**kwargs, stat='density', bins=list(self.bin_edges[param]), kde=plot_kde1d)

                if plot_prior:
                    sns.histplot(**histkwargs, **prior1d_kwargs)
                sns.histplot(**histkwargs, **post1d_kwargs, weights=ratio)

            elif plot_kde1d:
                kdekwargs = dict(**kwargs, cut=cut)
                if plot_prior:
                    sns.kdeplot(**kdekwargs, **prior1d_kwargs)
                sns.kdeplot(**kdekwargs, **post1d_kwargs, weights=ratio)

            if plot_bounds:
                ax.axvspan(self.ranges[param][0], bounds[param][0], **bounds_kwargs)
                ax.axvspan(bounds[param][1], self.ranges[param][1], **bounds_kwargs)

            ax.legend()

        for (i1, param1), (i2, param2) in combinations(enumerate(group), 2):
            ax: plt.Axes = axs[i2, i1]

            kwargs = dict(x=self.samples[param1], y=self.samples[param2])

            if plot_hist2d:
                histkwargs = dict(**kwargs, bins=(self.bin_edges[param1], self.bin_edges[param2]), density=True)
                hist_prior, *_ = np.histogram2d(**histkwargs)
                hist_post, xedges, yedges = np.histogram2d(**histkwargs, weights=ratio)
                imkwargs = dict(extent=(*xedges[(0, -1),], *yedges[(0, -1),]), origin='lower', aspect='auto')

                if plot_prior:
                    ax.imshow(hist_prior.T, **{**imkwargs, **dict(alpha=(hist_prior.max()/hist_post.max()).clip(0, 1), cmap=prior_cmap), **prior2d_kwargs})
                ax.imshow(hist_post.T, **{**imkwargs, **dict(cmap=post_cmap), **post2d_kwargs})

            if plot_kde2d:
                kdekwargs = dict(**kwargs, cut=cut, levels=[1-l for l in sorted(levels, reverse=True)], ax=ax)
                if plot_prior:
                    sns.kdeplot(**{**kdekwargs, **dict(color=prior_color), **prior2d_kwargs})
                sns.kdeplot(**{**kdekwargs, **dict(color=post_color), **post2d_kwargs}, weights=ratio)

            if plot_bounds:
                ax.axvspan(self.ranges[param1][0], bounds[param1][0], **bounds_kwargs)
                ax.axvspan(bounds[param1][1], self.ranges[param1][1], **bounds_kwargs)
                ax.axhspan(self.ranges[param2][0], bounds[param2][0], **bounds_kwargs)
                ax.axhspan(bounds[param2][1], self.ranges[param2][1], **bounds_kwargs)

            if plot_truth:
                truth_x, truth_y = (self.truths.get(key, None) for key in (param1, param2))

                if truth2d_type == 'lines':
                    if truth_x is not None:
                        ax.axvline(truth_x, color=truth_color)
                    if truth_y is not None:
                        ax.axhline(truth_y, color=truth_color)
                if truth2d_type == 'marker' and None not in (truth_x, truth_y):
                    ax.plot(self.truths[param1], self.truths[param2], truth_marker, color=truth_color, label='truth')

        return fig, axs

    # end CORNER

    # LOCAL

    @dataclass(eq=False)
    class Stats:
        d: DatasetWeighted
        sample_dims: tuple[_KT, ...]

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

    def stats(self, param_name: _KT, cred: _CredT = None) -> Stats:
        return self.Stats(self._weighted(param_name), self.sample_dims, cred)

    def _local_v_truth(self, param_name: _KT, cred: _CredT, ax: plt.Axes, **kwargs) -> plt.Axes:
        s = self.stats(param_name, cred)

        ax.errorbar(
            self.truths[param_name], s.means[param_name],
            ((s.meds-s.los)[param_name], (s.his-s.meds)[param_name]),
            **{**dict(
                color=self.post_color,
                ls='none', marker='.', markersize=5, markeredgecolor='none',
                elinewidth=0.5, capsize=2, capthick=0.5
            ), **kwargs}
        )

        return ax

    def local_v_truth(self, param_name: _KT, cred: float, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        if ax is None:
            ax = plt.gca()

            ax.plot(*2*((self.truths[param_name].min(), self.truths[param_name].max()),), 'k--')
            ax.set_xlabel('true')
            ax.set_ylabel('inferred')
            ax.set_title(self.param_label(param_name))

        return self._local_v_truth(param_name, cred, ax, **kwargs)

    # end LOCAL


@attr.define(slots=False, getstate_setstate=True)
class MultiLatentMomentComparisonPlotter(MultiSBIPosteriorPlotter):
    xstats: Mapping[_KT, MultiSBIPosteriorPlotter.Stats] = attr.ib(kw_only=True)

    meanclr = 'C0'
    stdclr = 'C1'

    diag_kwargs: Mapping[str, Any] = attr.field(default={}, converter=dict(color='k', ls='--').__or__)
    label_kwargs: Mapping[str, Any] = attr.field(default={}, converter=dict(labelsize='x-small').__or__)
    marker_kwargs: Mapping[str, Any] = attr.field(default={}, converter=dict(alpha=0.5, mew=0).__or__)

    def make_moment_axes(self, ax: plt.Axes) -> tuple[plt.Axes, plt.Axes]:
        ax.text(0.5, 0.02, 'mean', color=self.meanclr, ha='center', va='bottom', transform=ax.transAxes)
        ax.text(0.5, 0.98, 'st.dev.', color=self.stdclr, ha='center', va='top', transform=ax.transAxes)

        ax.plot([0, 1], [0, 1], transform=ax.transAxes, **self.diag_kwargs)

        meanax = ax
        stdax = ax.figure.add_axes(ax.get_position())
        stdax.patch.set_visible(False)

        meanax.tick_params(colors=self.meanclr, **self.label_kwargs)

        meanax.spines.right.set_visible(False)
        meanax.spines.top.set_visible(False)
        meanax.spines.left.set_color(self.meanclr)
        meanax.spines.bottom.set_color(self.meanclr)

        stdax.spines.left.set_visible(False)
        stdax.spines.bottom.set_visible(False)
        stdax.spines.right.set_color(self.stdclr)
        stdax.spines.top.set_color(self.stdclr)

        stdax.tick_params(colors=self.stdclr, **self.label_kwargs)
        stdax.xaxis.tick_top()
        stdax.xaxis.set_label_position('top')
        stdax.yaxis.tick_right()
        stdax.yaxis.set_label_position('right')
        stdax.yaxis.label.set_va('bottom')
        stdax.yaxis.label.set_rotation(-90)

        return meanax, stdax

    def moment_plot(self, param_name: _KT, ax: plt.Axes) -> tuple[plt.Axes, plt.Axes]:
        ax.text(0.02, 0.98, self.param_label(param_name), ha='left', va='top', transform=ax.transAxes)
        meanax, stdax = self.make_moment_axes(ax)

        ystats = self.Stats(self._weighted(param_name), self.sample_dims)
        for ax, stat, clr in ((meanax, 'mean', self.meanclr), (stdax, 'std', self.stdclr)):
            x, y = (getattr(stats, stat+'s')[param_name] for stats in (self.xstats[param_name], ystats))
            ax.plot(x, y, '.', color=clr, label=stat, **self.marker_kwargs)

            mmin, mmax = min(x.min(), y.min()), max(x.max(), y.max())

            ax.set_xlim(mmin - 0.1*(mmax-mmin), mmax + 0.1*(mmax-mmin))
            ax.set_ylim(*ax.get_xlim())

            ax.margins(0.1, 0.1, tight=True)
            # ax.set_aspect('equal', adjustable='datalim')

            ax.set_xticks(ax.get_xticks())
            ax.set_yticks(ax.get_xticks())

        return meanax, stdax



@attr.define
class MultiSBIValidationPlotter(MultiSBIPlotter):
    pp_xlabel: str = 'nominal credibility'
    pp_ylabel: str = 'empirical coverage'

    pp_diag_kwargs: Mapping = attr.field(default={}, converter=dict(color='black').__or__)

    def pp(
        self, creds: _MultiMappingT, ax: plt.Axes = None, *,
        flip_axes=False, sigmas=False, **kwargs
    ) -> plt.Axes:
        if (_ax := ax) is None:
            ax = plt.gca()

            # TODO: diagonal line for sigmas
            ax.plot(*2*((0, 3 if sigmas else 1),), **self.pp_diag_kwargs)

            (ax.set_ylabel if flip_axes else ax.set_xlabel)(self.pp_xlabel)
            (ax.set_xlabel if flip_axes else ax.set_ylabel)(self.pp_ylabel)

            if sigmas:
                ax.set_xlabel(ax.get_xlabel() + ' (sigmas)')
                ax.set_ylabel(ax.get_ylabel() + ' (sigmas)')
            else:
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

            ax.set_aspect('equal')

        for key, qs in creds.items():
            xy = sorted(qs), np.linspace(0, 1, len(qs))
            ax.plot(*(map(halfnorm().ppf, xy) if sigmas else xy)[::-1 if flip_axes else 1],
                    label=self.group_label(key), **kwargs)

        if _ax is None:
            ax.legend()

        return ax

    # def plot_norms(self, norms, fig=None, axis='posterior') -> plt.Figure:
    #     if fig is None:
    #         fig = plt.figure()
    #
    #     plt.axhline(1., color='k')
    #     try:
    #         sns.violinplot(
    #             data=list(norms.values()),
    #             cut=0., inner='box', saturation=1., scale='width'
    #         ).set_xticklabels(tuple(map(self.nrep.group_label, norms.keys())))
    #     except ValueError:
    #         pass
    #     plt.suptitle(f'NRE {axis} normalisation')
    #     plt.ylim(0.5, 1.5)
    #
    #     return fig
