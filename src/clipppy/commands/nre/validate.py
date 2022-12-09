from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from operator import itemgetter
from typing import Callable, Iterable, Mapping, Optional, Sequence

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from tqdm.auto import tqdm, trange

from ...sbi._typing import SBIDataset
from ...utils import _KT, _T, _VT
from ...utils.plotting.nre import _HeadT, _TailT, BaseNREPlotter, MultiNREPlotter, NREPlotter


def get_n_batches(dataset: Iterable[_T], n: int, *funcs: Callable[[_T], _VT]) -> Sequence[list[_VT]]:
    while True:
        ress = tuple(list() for _ in funcs)
        for _, batch in zip(trange(n, leave=False), dataset):
            for res, func in zip(ress, funcs):
                res.append(func(batch))
        yield ress


def get_n_batches_mapping(dataset: Iterable[_T], nbatches: int, *funcs: Callable[[_T], Mapping[_KT, _VT]]) -> Sequence[Mapping[_KT, list[_VT]]]:
    while True:
        ress = tuple(defaultdict(list) for _ in funcs)
        for _, batch in zip(trange(nbatches), dataset):
            for res, func in zip(ress, funcs):
                for key, val in func(batch).items():
                    res[key].append(val)
        yield ress


@dataclass
class BaseNREValidator:
    nbatches: int
    batch_size: int

    dataset: Optional[SBIDataset]
    nrep: BaseNREPlotter

    def __post_init__(self):
        if isinstance(self.dataset, SBIDataset):
            self.dataset.dataset.batch_size = self.batch_size

    def _simulate(self, head: _HeadT, tail: _TailT, ranger: Callable[[int], Iterable] = trange):
        for _, (params, obs) in zip(ranger(self.nbatches), self.dataset):
            yield params, self.nrep.ratio(obs, head, tail)

    @staticmethod
    def _simulate_finalize(qs, ratios, posts, prior_mean):
        return (
            torch.cat(qs, 0).tolist(),
            torch.cat(ratios, 0).mean(0).rename_(None).flatten().tolist(),
            (torch.cat(posts, 0).rename_(None).flatten(start_dim=1).mean(-1) / prior_mean).tolist()
        )

    def simulate(self, head: _HeadT, tail: _TailT, ranger: Callable[[int], Iterable] = trange):
        return self._simulate_finalize(*zip(*(
            (self.nrep.perc(params, post), ratio, post)
            for params, ratio in self._simulate(head, tail, ranger)
            for post in [self.nrep._post(ratio)]
        )), self.nrep.prior_mean)

    def plot_qq(self, qs, fig=None, *, flip_axes=False, sigmas=False) -> plt.Figure:
        if fig is None:
            fig = plt.figure()
        ax = self.nrep.qq(qs, flip_axes=flip_axes, sigmas=sigmas)
        ax.legend()
        ax.set_aspect('equal')
        return fig

    def plot_norms(self, norms, fig=None, axis='posterior') -> plt.Figure:
        if fig is None:
            fig = plt.figure()

        plt.axhline(1., color='k')
        try:
            sns.violinplot(
                data=list(norms.values()),
                cut=0., inner='box', saturation=1., scale='width'
            ).set_xticklabels(tuple(map(self.nrep.group_label, norms.keys())))
        except ValueError:
            pass
        plt.suptitle(f'NRE {axis} normalisation')
        plt.ylim(0.5, 1.5)

        return fig

    def __call__(self, head: _HeadT, tail: _TailT, *args, **kwargs):
        qs, norms_like, norms_post = self.simulate(head, tail, *args, **kwargs)
        return (
            self.plot_qq(qs),
            self.plot_norms(norms_like, axis='likelihood'),
            self.plot_norms(norms_post, axis='posterior')
        )


@dataclass
class BaseMultiNREValidator(BaseNREValidator):
    nrep: MultiNREPlotter


class BaseNRECoverage(ABC):
    @abstractmethod
    def confidence(self, head: _HeadT, tail: _TailT) -> Tensor: ...


@dataclass
class NRECoverage(BaseNRECoverage, BaseNREValidator):
    nrep: NREPlotter

    def confidence(self, head: _HeadT, tail: _TailT):
        qs = []
        for pots, in tqdm(self.dataset.dataset.conditioner_cls(
            get_n_batches(self.dataset, self.nbatches, partial(
                self.nrep.percentile_of_truth,
                head=head,
                tail=tail
            )), {
                key: val.unsqueeze(-1).expand(*val.shape, self.batch_size)
                for key, val in self.nrep.grid.items()
            }
        ), total=self.nrep.grid_shape.numel()):
            qs.append(torch.cat(pots))
        return torch.stack(qs, 0).unflatten(0, self.nrep.grid_shape)


class MultiNRECoverage(BaseNRECoverage, BaseMultiNREValidator):
    coverages: Mapping[str, NRECoverage] = field(init=False)

    def __post_init__(self):
        self.coverages = {group: NRECoverage(
            self.nbatches, self.batch_size,
            dataset=self.dataset, nrep=plotter
        ) for group, plotter in self.nrep.plotters.items()}

    def confidence(self, head: _HeadT, tail: _TailT):
        return {group: self.coverages[group].confidence(head, tail)
                for group, tail in self.nrep.subtails(tail).items()}


class MultiNREValidator(BaseMultiNREValidator):
    def _simulate_finalize(self, qs, ratios, posts, prior_mean):
        _qs, _ratios, _posts = {}, {}, {}
        for group in self.nrep.plotters.keys():
            _qs[group], _ratios[group], _posts[group] = super()._simulate_finalize(*(
                list(map(itemgetter(group), _)) for _ in (qs, ratios, posts)
            ), prior_mean[group])
        return _qs, _ratios, _posts
    # def simulate(self, head: _HeadT, tail: _TailT, ranger: Callable[[int], Iterable] = trange):
    #     qs = defaultdict(list)
    #     norms = defaultdict(list)
    #     for params, post in self._simulate(head, tail, ranger):
    #         for key, val in self.nrep.perc(params, post).items():
    #             qs[key] += val.tolist()
    #         for key, val in post.items():
    #             norms[key] += (val.mean(key)).tolist()
    #
    #     return qs, pd.DataFrame(norms).mul(pd.Series({g: plotter.range_area for g, plotter in self.nrep.plotters.items()}), axis='columns')
