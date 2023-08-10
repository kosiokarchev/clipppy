from abc import ABC, abstractmethod
from collections import ChainMap
from functools import cached_property
from itertools import combinations
from math import log, inf
from operator import itemgetter
from typing import Mapping, Sequence, Union, Iterable, Any, Literal, cast

import attr
import numpy as np
import seaborn as sns
import torch
from frozendict import frozendict
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from more_itertools import always_iterable
from scipy.stats import halfnorm
from torch import Tensor
from torch.nn import Module
from typing_extensions import TypeAlias

from .. import to_percentiles
from ....sbi._typing import SBIBatch, _SBIObsT
from ....sbi.nn import BaseSBIHead


_HeadT: TypeAlias = BaseSBIHead
_TailT: TypeAlias = Module
_MultiTailKT: TypeAlias = Union[str, Iterable[str]]

_qT: TypeAlias = Sequence

_ColorT: TypeAlias = Any
_MarkerT: TypeAlias = str


@attr.s(auto_attribs=True, kw_only=True)
class BaseNREPlotter(ABC):
    params: Mapping[str, Tensor] = attr.ib(repr=False)
    _param_names: Sequence[str] = None

    @property
    def param_names(self) -> Sequence[str]:
        if self._param_names is None:
            self._param_names = tuple(self.params.keys())
        return self._param_names

    @cached_property
    def nparams(self) -> int:
        return len(self.param_names)

    @cached_property
    def paramsnd(self):
        return torch.stack(itemgetter(*self.param_names)(self.params), -1)

    @cached_property
    def ranges(self) -> Mapping[str, tuple[float, float]]:
        return {key: (self.params[key].min().item(), self.params[key].max().item()) for key in self.param_names}

    labels: Mapping[str, str] = attr.ib(factory=dict)

    def _param_label(self, param_name) -> str:
        return self.labels.get(param_name, param_name)

    def group_label(self, g) -> str:
        return ', '.join(map(self._param_label, always_iterable(g)))

    @abstractmethod
    def log_ratio(self, obs: _SBIObsT, head: _HeadT, tail: _TailT, params=None): ...

    @abstractmethod
    def ratio(self, obs: _SBIObsT, head: _HeadT, tail: _TailT, params=None): ...

    @abstractmethod
    def _perc(self, log_ratios, log_ratio): ...

    def perc(self, params: Mapping[str, Tensor], obs: _SBIObsT, head: _HeadT, tail: _TailT, log_ratio=None):
        return self._perc(
            self.log_ratio(obs, head, tail, params),
            self.log_ratio(obs, head, tail, self.params)
            if log_ratio is None else log_ratio
        )

    _qq_xlabel: str = 'nominal coverage'
    _qq_ylabel: str = 'empirical coverage'

    def qq(self, *args, ax: plt.Axes = None, truth_kwargs=frozendict(), flip_axes: bool = False, sigmas: bool = False, **kwargs):
        if ax is None:
            ax: plt.Axes = plt.gca()

            # TODO: diagonal line for sigmas
            ax.plot(*2*((0, 3 if sigmas else 1),), **{**dict(color='black'), **truth_kwargs})

            (ax.set_ylabel if flip_axes else ax.set_xlabel)(self._qq_xlabel)
            (ax.set_xlabel if flip_axes else ax.set_ylabel)(self._qq_ylabel)

            if sigmas:
                ax.set_xlabel(ax.get_xlabel() + ' (sigmas)')
                ax.set_ylabel(ax.get_ylabel() + ' (sigmas)')
            else:
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

            ax.set_aspect('equal')

        return ax


@attr.s
class NREPlotter(BaseNREPlotter):
    @cached_property
    def bin_edges(self) -> Mapping[str, np.ndarray]:
        return {
            param: np.histogram_bin_edges(self.params[param].numpy(), 'auto')
            for param in self.param_names
        }

    def log_ratio(self, obs: _SBIObsT, head: _HeadT, tail: _TailT, params=None) -> Tensor:
        head.eval(), tail.eval()
        with torch.no_grad():
            return tail(*head(params if params is not None else self.params, obs))

    def ratio(self, obs: _SBIObsT, head: _HeadT, tail: _TailT, params=None) -> Tensor:
        return self.log_ratio(obs, head, tail, params).exp_()

    def _perc(self, log_ratios, log_ratio) -> Tensor:
        return (torch.where(
            cast(Tensor, log_ratios.unsqueeze(-1) < log_ratio.unsqueeze(-2)),
            log_ratio.new_full((), -inf), log_ratio
        ).logsumexp(-1) - log(log_ratio.shape[-1])).exp_()

    def percentile_of_truth(self, trace: SBIBatch, head: _HeadT, tail: _TailT):
        log_ratio_truth = self.log_ratio(trace.obs, head, tail, trace.params).unsqueeze(-1)
        log_ratios = self.log_ratio({
            key: val.unsqueeze(-head.event_dims.get(key, val.ndim)-1)
            for key, val in trace.obs.items()
        }, head, tail)

        return (log_ratios.where(log_ratios > log_ratio_truth, log_ratios.new_full((), -inf)).logsumexp(-1) - log_ratios.logsumexp(-1)).exp_()

    prior_kwargs = dict(label='prior')
    prior1d_kwargs = dict()
    prior2d_kwargs = dict()
    post_kwargs = dict(label='posterior')
    post1d_kwargs = dict()
    post2d_kwargs = dict()

    bounds_kwargs = dict(color='0.8', zorder=-1)

    def corner(
        self, ratio: Tensor, *,
        plot_prior=True, plot_hist1d=True, plot_hist2d=True, plot_kde1d=True, plot_kde2d=True, plot_bounds=True,
        prior_color: _ColorT = 'C1', prior_kwargs=frozendict(), prior1d_kwargs=frozendict(), prior2d_kwargs=frozendict(),
        post_color: _ColorT = 'C0', post_kwargs=frozendict(), post1d_kwargs=frozendict(), post2d_kwargs=frozendict(),
        levels: Sequence[float] = (0.68, 0.95, 1-1e-4), cut=True,
        truths: Mapping[str, float] = frozendict(),
        truth2d_type: Literal['lines', 'marker'] = 'lines',
        truth_marker: _MarkerT = '*',
        truth_color: _ColorT = 'green',
        bounds_method_kwargs=frozendict(),
        bounds_kwargs=frozendict(),
        figsize=None
    ) -> tuple[plt.Figure, Union[np.ndarray, Sequence[Sequence[plt.Axes]]]]:
        if plot_bounds:
            bounds = self.bounds(ratio, **bounds_method_kwargs)

        ratio = ratio.numpy(force=True)

        prior_kwargs = ChainMap(self.prior_kwargs, prior_kwargs)
        prior1d_kwargs = ChainMap(prior_kwargs, self.prior1d_kwargs, dict(color=prior_color), prior1d_kwargs)
        prior2d_kwargs = ChainMap(prior_kwargs, self.prior2d_kwargs, dict(cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list('cmap_prior', ((1, 1, 1, 0), prior_color))), prior2d_kwargs)

        post_kwargs = ChainMap(self.post_kwargs, post_kwargs)
        post1d_kwargs = ChainMap(post_kwargs, self.post1d_kwargs, dict(color=post_color), post1d_kwargs)
        post2d_kwargs = ChainMap(post_kwargs, self.post2d_kwargs, dict(cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list('cmap_post', ((1, 1, 1, 0), post_color))), post2d_kwargs)

        bounds_kwargs = ChainMap(self.bounds_kwargs, bounds_kwargs)

        if figsize is None:
            figsize = 3 * np.array(2 * (len(self.param_names),)) + 1

        fig, axs = plt.subplots(self.nparams, self.nparams, figsize=figsize, squeeze=False)

        axs = np.atleast_2d(axs)

        for ax in axs.flatten():
            ax.set_axis_off()
        for ax in axs[:-1].flatten():
            ax.set_xticklabels([])
        for ax in axs[:, 1:].flatten():
            ax.set_yticklabels([])

        for i, param in enumerate(self.param_names):
            if i:
                axs[i, 0].set_ylabel(self._param_label(param))
            axs[-1, i].set_xlabel(self._param_label(param))

            ax: plt.Axes = axs[i, i]
            ax.set_axis_on()
            ax.set_yticklabels([])

            if (truth := truths.get(param, None)) is not None:
                ax.axvline(truth, color=truth_color, label='truth')

            kwargs = dict(x=self.params[param], ax=ax)

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

        for (i1, param1), (i2, param2) in combinations(enumerate(self.param_names), 2):
            ax: plt.Axes = axs[i2, i1]
            ax.set_axis_on()

            kwargs = dict(x=self.params[param1].numpy(force=True), y=self.params[param2].numpy(force=True))

            if plot_hist2d:
                histkwargs = dict(**kwargs, bins=(self.bin_edges[param1], self.bin_edges[param2]), density=True)
                hist_prior, *_ = np.histogram2d(**histkwargs)
                hist_post, xedges, yedges = np.histogram2d(**histkwargs, weights=ratio)
                imkwargs = dict(extent=(*xedges[(0, -1),], *yedges[(0, -1),]), origin='lower')

                if plot_prior:
                    ax.imshow(hist_prior.T, **imkwargs, **{**dict(alpha=(hist_prior.max()/hist_post.max()).clip(0, 1)), **prior2d_kwargs})
                ax.imshow(hist_post.T, **imkwargs, **post2d_kwargs)

            if plot_kde2d:
                kdekwargs = dict(**kwargs, cut=cut, levels=[1-l for l in reversed(levels)], ax=ax)
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

            truth_x, truth_y = (truths.get(key, None) for key in (param1, param2))

            if truth2d_type == 'lines':
                if truth_x is not None:
                    ax.axvline(truth_x, color=truth_color)
                if truth_y is not None:
                    ax.axhline(truth_y, color=truth_color)
            if truth2d_type == 'marker' and None not in (truth_x, truth_y):
                ax.plot(truths[param1], truths[param2], truth_marker, color=truth_color)

        return fig, axs

    def qq(
        self, qs: _qT = (), *,
        ax: plt.Axes = None, truth_kwargs=frozendict(),
        flip_axes: bool = False, sigmas: bool = False,
        **kwargs
    ):
        ax = super().qq(ax=ax, truth_kwargs=truth_kwargs, flip_axes=flip_axes, sigmas=sigmas)
        xy = sorted(qs), np.linspace(0, 1, len(qs))
        ax.plot(*(map(halfnorm().ppf, xy) if sigmas else xy)[::-1 if flip_axes else 1], **kwargs)

        return ax

    @staticmethod
    def mask_from_postmass(ratio: Tensor, thresh=1e-4):
        return to_percentiles(ratio) < 1-thresh

    @staticmethod
    def mask_from_like_ratio(ratio: Tensor, thresh=1e-4):
        return ratio/ratio.max() > thresh

    def bounds(self, ratio: Tensor, thresh=1e-4, method: Literal['postmass', 'like_ratio'] = 'postmass') -> Mapping[str, tuple[float, float]]:
        mask = (self.mask_from_postmass if method == 'postmass' else self.mask_from_like_ratio)(ratio, thresh)
        return {key: (
            val.where(mask, val.new_tensor([inf]).expand_as(val)).amin(-1).item(),
            val.where(mask, val.new_tensor([-inf]).expand_as(val)).amax(-1).item(),
        ) for key in self.param_names for val in [self.params[key]]}
