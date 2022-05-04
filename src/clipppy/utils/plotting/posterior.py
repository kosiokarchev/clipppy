from __future__ import annotations

from functools import cached_property
from itertools import combinations
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
from frozendict import frozendict
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap
from torch import Tensor
from typing_extensions import TypeAlias

from .nre import to_percentiles


_truthT: TypeAlias = float
_truthsT: TypeAlias = Mapping[str, float]

_axsT: TypeAlias = Union[np.ndarray, Sequence[Sequence[plt.Axes]]]
_ColorT: TypeAlias = Any
_LineT: TypeAlias = str
_MarkerT: TypeAlias = str
_CmapT: TypeAlias = Union[str, Colormap]

_empty_dict: Mapping = frozendict()

post_cmap = LinearSegmentedColormap.from_list('post', (
    (1, 1, 1, 0),
    (1, 0, 0, 1),
    (1, 1, 0, 1),
    (0, 1, 0, 1)
))


class PosteriorPlotter:
    corner_names: Sequence[str]

    labels: Mapping[str, str]

    def _label(self, name):
        return self.labels.get(name, name)

    ranges: Mapping[str, tuple[float, float]]

    def _range(self, name):
        return self.ranges.get(name, (None, None))

    @cached_property
    def ncorner(self):
        return len(self.corner_names)

    @property
    def iter_1d(self) -> Iterable[tuple[int, str]]:
        return enumerate(self.corner_names)

    @property
    def iter_2d(self) -> Iterable[tuple[tuple[int, str], tuple[int, str]]]:
        return combinations(enumerate(self.corner_names), 2)

    def _setup_1dax(self, ax: plt.Axes, param_name=None):
        ax.set_yticklabels([])
        ax.set_xlim(*self._range(param_name))
        ax.set_ylim(bottom=0, auto=True)

    def _setup_2dax(self, ax: plt.Axes, paramx, paramy):
        ax.set_xlim(*self._range(paramx))
        ax.set_ylim(*self._range(paramy))

    def _setup_axes(self, axs: _axsT):
        for i in range(axs.shape[1]-1):
            for j in range(1, axs.shape[0]):
                axs[i, j].set_axis_off()

        for i, param_name in self.iter_1d:
            self._setup_1dax(axs[i, i], param_name)
        for (ix, paramx), (iy, paramy) in self.iter_2d:
            self._setup_2dax(axs[iy, ix], paramx, paramy)

        for ax in axs[:-1].flatten():
            ax.set_xticklabels([])
        for ax in axs[:, 1:].flatten():
            ax.set_yticklabels([])

        for i, param_name in self.iter_1d:
            if i:
                axs[i, 0].set_ylabel(self._label(param_name))
            axs[-1, i].set_xlabel(self._label(param_name))

        return axs

    def _corner_figure(self, fig: plt.Figure = None, figure_kwargs=_empty_dict) -> tuple[plt.Figure, _axsT]:
        if fig is None:
            fig, axs = plt.subplots(*2*(self.ncorner,), squeeze=False, **{**self.default_figure_kwargs, **figure_kwargs})
        else:
            axs = np.array(fig.axes).reshape(2*(self.ncorner,))
        return fig, axs

    def plot_truth_1d(self, ax: plt.Axes, truth: _truthT, truth1d_kwargs):
        if truth is not None:
            return ax.axvline(truth, **{
                'color': self.default_truth_color,
                **self.default_truth1d_kwargs, **truth1d_kwargs})

    def plot_truth_2d(self, ax: plt.Axes, truthx: _truthT, truthy: _truthT,
                      line_kwargs=_empty_dict, marker_kwargs=_empty_dict) -> tuple[Optional[plt.Line2D], Optional[plt.Line2D], Optional[plt.Line2D]]:
        line_kwargs = {
            'color': self.default_truth_color,
            **self.default_truth_line_kwargs, **line_kwargs}
        marker_kwargs = {
            'color': self.default_truth_color,
            **self.default_truth_marker_kwargs, **marker_kwargs}
        return (
            ax.axvline(truthx, **line_kwargs) if line_kwargs['linestyle'] and truthx is not None else None,
            ax.axhline(truthy, **line_kwargs) if line_kwargs['linestyle'] and truthy is not None else None,
            ax.plot(truthx, truthy, **marker_kwargs) if marker_kwargs['marker'] and truthx is not None and truthy is not None else None
        )

    def plot_truths(self, axs: _axsT, truths: _truthsT,
                    truth1d_kwargs=_empty_dict, line_kwargs=_empty_dict, marker_kwargs=_empty_dict):
        for i, param_name in self.iter_1d:
            self.plot_truth_1d(axs[i, i], truths.get(param_name, None), truth1d_kwargs)
        for (ix, paramx), (iy, paramy) in self.iter_2d:
            self.plot_truth_2d(axs[iy, ix], truths.get(paramx, None), truths.get(paramy, None),
                               line_kwargs, marker_kwargs)

    def plot_line1d(self, ax: plt.Axes, x=None, y=None, line1d_kwargs=_empty_dict):
        return ax.plot(x, y / ((y[1:] + y[:-1]) / 2 * np.diff(x)).sum(), **{**self.default_line1d_kwargs, **line1d_kwargs})

    def plot_img2d(self, ax: plt.Axes, img: np.ndarray, extent, img_kwargs):
        ax.imshow(img, **{**self.default_img_kwargs, **img_kwargs}, extent=extent)

    def plot_contour2d(self, ax: plt.Axes, img: np.ndarray, extent, contour_kwargs=_empty_dict):
        contour_kwargs = {**self.default_contour_kwargs, **contour_kwargs}
        if contour_kwargs.get('levels', None):
            return ax.contour(img, **contour_kwargs, extent=extent)

    @property
    def default_figure_kwargs(self):
        return dict(figsize=3 * np.array(2*(self.ncorner,)) + 1)

    default_truth_color = None
    default_truth1d_kwargs = dict(linestyle='--')
    default_truth_line_kwargs = dict(linestyle=None)
    default_truth_marker_kwargs = dict(marker='*')

    default_line1d_kwargs = dict()
    default_contour_kwargs = dict(origin='lower', levels=(0.68, 0.95), colors='black', linestyles=('-', '--'))
    default_img_kwargs = dict(origin='lower', vmin=0, cmap=post_cmap)

    def corner(
        self, *args, fig: plt.Figure = None, figure_kwargs=_empty_dict,
        contour_kwargs=_empty_dict,
        truths: _truthsT = _empty_dict, truths_line_kwargs=_empty_dict, truths_marker_kwargs=_empty_dict,
        **kwargs
    ):
        return self._corner_figure(fig, figure_kwargs)


class FuncPosteriorPlotter(PosteriorPlotter):
    def corner(
        self,
        grids: Mapping[str, Tensor] = _empty_dict, post: Tensor = None, cmap: _CmapT = post_cmap,
        *args,
        fig: plt.Figure = None, figure_kwargs=_empty_dict,
        line1d_kwargs=_empty_dict,
        plot_dist=True, img_kwargs=_empty_dict,
        contour_kwargs=_empty_dict,
        truths: _truthsT = _empty_dict, truths_line_kwargs=_empty_dict, truths_marker_kwargs=_empty_dict,
    ):
        fig, axs = super().corner(**{key: val for key, val in locals().items() if key != 'self'})

        available = grids.keys() & set(post.names)

        for i, param_name in self.iter_1d:
            if param_name in available:
                otherparams = tuple(set(self.corner_names) - {param_name})
                line = (post.mean(otherparams) if otherparams else post).rename(None)
                self.plot_line1d(axs[i, i], grids[param_name], line, line1d_kwargs)

        for (ix, paramx), (iy, paramy) in self.iter_2d:
            if paramx in available and paramy in available:
                otherparams = tuple(set(self.corner_names) - {paramx, paramy})
                img = (post.mean(otherparams) if otherparams else post).align_to(paramy, paramx).rename(None)
                extent = (*self.ranges[paramx], *self.ranges[paramy])
                if plot_dist:
                    self.plot_img2d(axs[iy, ix], img, extent, img_kwargs)
                self.plot_contour2d(axs[iy, ix], to_percentiles(img), extent, contour_kwargs)

        self.plot_truths(axs, truths, truths_line_kwargs, truths_marker_kwargs)

        return fig, self._setup_axes(axs)
