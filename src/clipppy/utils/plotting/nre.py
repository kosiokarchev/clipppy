from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property, partial, partialmethod, reduce
from itertools import chain, combinations
from math import inf
from operator import itemgetter, or_
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, Type, Union

import attr
import numpy as np
import pandas as pd
import torch
from frozendict import frozendict
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
from more_itertools import always_iterable, last, padded
from scipy.stats import halfnorm
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Module
from typing_extensions import TypeAlias

# TODO: dependency on phytorch, uplot
from phytorch.utils import _mid_many, ravel_multi_index
from uplot import imshow_with_cbar, midtraffic

from ...sbi.nn import BaseSBIHead, MultiSBITail
from ...sbi._typing import _SBIBatchT, MultiSBIProtocol


_HeadT: TypeAlias = BaseSBIHead
_TailT: TypeAlias = Module
_MultiTailKT: TypeAlias = Union[str, Iterable[str]]

_qT: TypeAlias = Sequence


_GridT: TypeAlias = Tensor
_ColorT: TypeAlias = Any
_MarkerT: TypeAlias = str
_CmapT: TypeAlias = Union[str, Colormap]


def to_percentiles(arr: Tensor, ndim=None):
    start_dim = arr.ndim - (ndim or arr.ndim)
    flatarr = arr.rename(None).flatten(start_dim)
    argsort = flatarr.argsort(-1, descending=True)
    return (
        torch.empty_like(flatarr, memory_format=torch.contiguous_format)
        .scatter_(
            -1, argsort,
            (flatarr.take_along_dim(argsort, -1).cumsum(-1) / flatarr.sum(-1, keepdim=True))
        ).unflatten(-1, arr.shape[start_dim:]).rename_(*arr.names)
    )


@attr.s(auto_attribs=True, kw_only=True)
class BaseNREPlotter(ABC):
    _param_names: Sequence[str] = None

    @property
    def param_names(self):
        if self._param_names is None:
            self._param_names = tuple(self.ranges.keys())
        return self._param_names

    @cached_property
    def nparams(self):
        return len(self.param_names)

    _ranges: Mapping[str, tuple[float, float]] = attr.ib(default=None, init=True)

    @property
    def ranges(self):
        if self._ranges is None:
            self._ranges = {key: (val[0].item(), val[-1].item()) for key, val in self.grids.items()}
        return self._ranges

    @cached_property
    def range_area(self) -> float:
        return np.product([rng[1]-rng[0] for key in self._param_names for rng in [self.ranges[key]]])

    grid_sizes: Mapping[str, int] = attr.ib(factory=partial(defaultdict, lambda: 64))

    _grids: Mapping[str, Tensor] = None

    @property
    def grids(self):
        if self._grids is None:
            self._grids = {
                key: _mid_many(torch.linspace(*self.ranges[key], self.grid_sizes[key]+1), (-1,))
                for key in self.param_names
            }
        return self._grids

    priors: Mapping[str, Distribution] = attr.ib(factory=dict)

    labels: Mapping[str, str] = attr.ib(factory=dict)

    def _param_label(self, param_name):
        return self.labels.get(param_name, param_name)

    def group_label(self, g):
        return ', '.join(map(self._param_label, always_iterable(g)))

    @property
    @abstractmethod
    def prior(self): ...

    @property
    @abstractmethod
    def prior_mean(self): ...

    @abstractmethod
    def log_ratio(self, obs, head: _HeadT, tail: _TailT): ...

    @abstractmethod
    def ratio(self, obs, head: _HeadT, tail: _TailT): ...

    @abstractmethod
    def _post(self, ratio): ...

    def post(self, obs, head: _HeadT, tail: _TailT):
        return self._post(self.ratio(obs, head, tail))

    @abstractmethod
    def perc(self, params: Mapping[str, Tensor], post: Tensor): ...

    @abstractmethod
    def get_bounds_from_post(self, post, thresh: float = 1e-4) -> Mapping[str, tuple[Tensor, Tensor]]: ...

    _qq_xlabel: str = 'nominal coverage'
    _qq_ylabel: str = 'empirical coverage'

    def qq(self, *args, ax: plt.Axes = None, truth_kwargs=frozendict(), flip_axes: bool = False, sigmas: bool = True, **kwargs):
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
    @staticmethod
    def mesh_grids(grids: Mapping[str, Tensor], names: Iterable[str]):
        return {
            key: val.reshape(-1)
            for key, val in zip(names, torch.meshgrid(*map(grids.get, names), indexing='ij'))
        }

    @cached_property
    def grid(self):
        return self.mesh_grids(self.grids, self.param_names)

    @cached_property
    def grid_shape(self):
        return torch.Size(map(self.grid_sizes.get, self.param_names))

    @cached_property
    def prior(self) -> Tensor:
        return sum((
            self.priors[key].log_prob(self.grids[key]).rename_(key).align_to(*self.param_names)
            for key in self.param_names if key in self.priors
        ), torch.tensor(0.).align_to(*self.param_names)).exp_()

    @cached_property
    def prior_mean(self) -> float:
        return self.prior.mean().item()

    def log_ratio(self, obs, head: _HeadT, tail: _TailT) -> Tensor:
        head.eval(), tail.eval()
        with torch.no_grad():
            return tail(*head(self.grid, {
                key: val.unsqueeze(-head.event_dims.get(key, val.ndim)-1)
                for key, val in obs.items()
            })).rename(..., 'grid').unflatten('grid', tuple(zip(self.param_names, self.grid_shape)))

    def ratio(self, obs, head: _HeadT, tail: _TailT) -> Tensor:
        # TODO: nan_to_num on named tensors
        return (res := self.log_ratio(obs, head, tail).exp_()).rename_(None).nan_to_num_().rename_(*res.names)

    def _post(self, ratio) -> Tensor:
        return self.prior * ratio

    def perc(self, params: Mapping[str, Tensor], post: Tensor) -> Tensor:
        return to_percentiles(post).rename(None).flatten(-self.nparams).take_along_dim(
            ravel_multi_index(tuple(
                torch.searchsorted(grid, params[param_name]).clamp_(0, len(grid)-1)
                for param_name in self.param_names for grid in [self.grids[param_name]]
            ), torch.Size(map(len, map(self.grids.get, self.param_names)))).unsqueeze(-1),
            -1
        ).squeeze(-1)

    def percentile_of_truth(self, trace: _SBIBatchT, head: _HeadT, tail: _TailT):
        return self.perc(trace[0], self.post(trace[1], head, tail))

    def get_bounds_from_post(self, post: Tensor, thresh: float = 1e-4) -> Mapping[str, tuple[Tensor, Tensor]]:
        mask = to_percentiles(post, len(post.names)).rename(None).flatten(-len(post.names)) < 1 - thresh
        return {key: (
            val.where(mask, val.new_tensor([inf]).expand_as(val)).amin(-1),
            val.where(mask, val.new_tensor([-inf]).expand_as(val)).amax(-1),
        ) for key in post.names for val in [self.grid[key]]}

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

    prior_kwargs = dict(ls='--', color='orange', label='prior')
    post1d_kwargs = dict(ls='-', color='black', label='posterior')

    def corner(
        self,
        post: Tensor,
        cmap: _CmapT = 'inferno',
        levels: Sequence[float] = (0.68, 0.95),
        levels_colors: Union[_ColorT, Sequence[_ColorT]] = ('red', 'green'),
        truths: Mapping[str, float] = frozendict(),
        truth2d_type: Literal['lines', 'marker'] = 'lines',
        truth_marker: _MarkerT = '*',
        truth_color: _ColorT = 'green',
        prior_kwargs=frozendict(), post1d_kwargs=frozendict(),
        figsize=None
    ):
        if figsize is None:
            figsize = 3 * np.array(2 * (len(self.param_names),)) + 1

        fig, axs = plt.subplots(self.nparams, self.nparams, figsize=figsize)

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
                ax.axvline(truth, color=truth_color)

            # TODO: Python 3.10: dict | frozendict
            for p, kw in ((self.prior, {**self.prior_kwargs, **prior_kwargs}), (post, {**self.post1d_kwargs, **post1d_kwargs})):
                ax.plot(self.grids[param], (
                    p.mean(tuple(set(self.param_names) - {param}))
                    if self.nparams > 1 else p
                ).rename(None), **kw)

            ax.set_xlim(*self.ranges[param])
            ax.set_ylim(bottom=0)

            ax.legend()

        for (i1, param1), (i2, param2) in combinations(enumerate(self.param_names), 2):
            ax: plt.Axes = axs[i2, i1]
            ax.set_axis_on()

            img = (
                post if post.ndim == 2 else
                post.mean(tuple(set(self.param_names) - {param1, param2}))
            ).align_to(param2, param1).rename(None)
            imkwargs = dict(origin='lower', extent=(*self.ranges[param1], *self.ranges[param2]))
            ax.imshow(img, **imkwargs, cmap=cmap, vmin=0, aspect='auto')
            ax.contour(to_percentiles(img), **imkwargs, levels=levels, colors=levels_colors)

            truth_x, truth_y = (truths.get(key, None) for key in (param1, param2))

            if truth2d_type == 'lines':
                if truth_x is not None:
                    ax.axvline(truth_x, color=truth_color)
                if truth_y is not None:
                    ax.axhline(truth_y, color=truth_color)
            if truth2d_type == 'marker' and None not in (truth_x, truth_y):
                ax.plot(truths[param1], truths[param2], truth_marker, color=truth_color)

        return fig, axs

    default_coverage2d_kwargs = dict(origin='lower', cmap=midtraffic)
    nominal_credibility_label = 'nominal credibility'
    coverage_title = 'Empirical coverage at {p:.1%}'

    def coverage2d(self, p, qs, ax=None, imshow_with_cbar_kwargs=frozendict(), **kwargs):
        if ax is None:
            ax = plt.gca()

        kwargs = {**self.default_coverage2d_kwargs, **kwargs}

        norm = kwargs.pop('norm', TwoSlopeNorm(p))

        cax = imshow_with_cbar(
            torch.quantile(qs, p, -1),
            extent=chain.from_iterable(itemgetter(*self.param_names)(self.ranges)),
            **kwargs, norm=norm, ax=ax, **imshow_with_cbar_kwargs
        )[1]

        cax.set_label(self.nominal_credibility_label)
        cax.ax.axhline(p, color='k')

        ax.title(self.coverage_title.format(p=p))
        ax.set_xlabel(self._param_label(self.param_names[0]))
        ax.set_ylabel(self._param_label(self.param_names[1]))

    def coverage(self, p, qs, ax=None, **kwargs):
        ax = ax or plt.gca()

        if len(self.param_names) == 2:
            self.coverage2d(p, qs, ax=ax, **kwargs)
        else:
            raise NotImplementedError

    def coverage_matrix(self, qs, masks=None, axs=None, plotsize=2, plotmargin=0.3):
        covdf = pd.DataFrame(
            qs.flatten(end_dim=-2).sort(-1).values,
            index=pd.MultiIndex.from_arrays(
                list(map(torch.Tensor.tolist, self.grid.values())),
                names=self.grid.keys())
        )

        idx = np.arange(self.grid_shape.numel()).reshape(self.grid_shape)[masks]

        if axs is None:
            axs = plt.subplots(
                *padded(idx.shape[::-1], 1, 2),
                sharex='all', sharey='all', squeeze=False,
                figsize=np.array(idx.shape) * plotsize + plotmargin
            )[1]

        for ax, (key, c) in zip(axs[::-1].T.flatten(), covdf.iloc[idx.flatten()].iterrows()):
            ax.plot(*2 * ((0, 1),), lw=0.5, color='k')
            ax.plot(c, np.linspace(0, 1, len(c)))

            lbls = [f'${self._param_label(k).strip("$")} = {v:.2f}$'
                    for k, v in zip(covdf.index.names, key)]

            ax.text(0.98, 0.02, lbls[0], transform=ax.transAxes, va='bottom', ha='right')
            if len(idx.shape) > 1:
                ax.text(0.02, 0.98, lbls[1], transform=ax.transAxes, va='top', ha='left')

        return axs


class PseudoTail(torch.nn.Module):
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
        BaseNREPlotter.log_ratio, BaseNREPlotter.ratio, BaseNREPlotter.post)):
    groups: Sequence[_MultiTailKT]
    subcls: Type[NREPlotter] = NREPlotter

    @cached_property
    def plotters(self) -> Mapping[_MultiTailKT, NREPlotter]:
        return {
            g: self.subcls(
                param_names=tuple(always_iterable(g)),
                # TODO: less verbose MultiNREPlotter
                ranges=self.ranges, grid_sizes=self.grid_sizes, grids=self.grids,
                priors=self.priors, labels=self.labels
            )
            for g in self.groups
        }

    def subtails(self, tail):
        return {g: PseudoTail(g, tail) for g in self.groups}

    def _mapped(self, funcname, obs, head: _HeadT, tail: MultiSBITail) -> Mapping[_MultiTailKT, Tensor]:
        return {g: getattr(self.plotters[g], funcname)(obs, head, subtail) for g, subtail in self.subtails(tail).items()}

    @property
    def prior(self) -> Mapping[_MultiTailKT, Tensor]:
        return {g: plotter.prior for g, plotter in self.plotters.items()}

    @cached_property
    def prior_mean(self) -> Mapping[_MultiTailKT, float]:
        return {g: plotter.prior_mean for g, plotter in self.plotters.items()}

    def _post(self, ratio: Mapping[_MultiTailKT, Tensor]) -> Mapping[_MultiTailKT, Tensor]:
        return {g: self.plotters[g]._post(r) for g, r in ratio.items()}

    def perc(self, params: Mapping[str, Tensor], post: Mapping[_MultiTailKT, Tensor]) -> Mapping[_MultiTailKT, Tensor]:
        return {g: self.plotters[g].perc(params, p) for g, p in post.items()}

    def get_bounds_from_post(self, post: Mapping[_MultiTailKT, Tensor], thresh: float = 1e-4) -> Mapping[str, tuple[Tensor, Tensor]]:
        return reduce(or_, (self.plotters[group].get_bounds_from_post(p, thresh) for group, p in post.items()))

    def qq(self, qs: Mapping[str, _qT] = (), ax: plt.Axes = None, **kwargs):
        ax = last(
            ax for ax in [super().qq(ax=ax, **kwargs)]
            for g, val in qs.items()
            for ax in [self.plotters[g].qq(val, label=self.group_label(g), ax=ax, **kwargs)]
        )
        ax.legend()
        return ax


def multi_posterior(nre: MultiSBIProtocol, nrep: MultiNREPlotter, trace):
    obs = {key: trace[key] for key in nre.obs_names}
    return {
        key: nrep.plotters[key].corner(
            nrep.plotters[key].post(obs, nre.head, subtail),
            truths={key: float(trace[key]) for key in nre.param_names
                    if not torch.is_tensor(trace[key]) or trace[key].numel() == 1}
        )[0]
        for key, subtail in nrep.subtails(nre.tail).items()
    }
