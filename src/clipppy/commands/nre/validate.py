from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from more_itertools import always_iterable
from torch import Tensor
from tqdm.auto import tqdm, trange

from ._typing import NREDataset
from ...utils import _KT, _T, _VT
from ...utils.plotting.nre import _HeadT, _TailT, MultiNREPlotter, NREPlotter


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

    dataset: NREDataset

    def __post_init__(self):
        self.dataset.dataset.batch_size = self.batch_size


@dataclass
class BaseMultiNREValidator(BaseNREValidator):
    nrep: MultiNREPlotter


class BaseNRECoverage(ABC):
    @abstractmethod
    def confidence(self, head: _HeadT, tail: _TailT) -> Tensor: ...


@dataclass
class NRECoverage(BaseNRECoverage, BaseNREValidator):
    plotter: NREPlotter

    def confidence(self, head: _HeadT, tail: _TailT):
        qs = []
        for pots, in tqdm(self.dataset.dataset.conditioner_cls(
            get_n_batches(self.dataset, self.nbatches, partial(
                self.plotter.percentile_of_truth,
                head=head,
                tail=tail
            )), {
                key: val.unsqueeze(-1).expand(*val.shape, self.batch_size)
                for key, val in self.plotter.grid.items()
            }
        ), total=self.plotter.grid_shape.numel()):
            qs.append(torch.cat(pots))
        return torch.stack(qs, 0).unflatten(0, self.plotter.grid_shape)


class MultiNRECoverage(BaseNRECoverage, BaseMultiNREValidator):
    coverages: Mapping[str, NRECoverage] = field(init=False)

    def __post_init__(self):
        self.coverages = {group: NRECoverage(
            self.nbatches, self.batch_size, dataset=self.dataset, plotter=plotter
        ) for group, plotter in self.nrep.plotters.items()}

    def confidence(self, head: _HeadT, tail: _TailT):
        return {group: self.coverages[group].confidence(head, tail)
                for group, tail in self.nrep.subtails(tail).items()}


class MultiNREValidator(BaseMultiNREValidator):
    def simulate(self, head: _HeadT, tail: _TailT):
        qs = defaultdict(list)
        norms = defaultdict(list)
        for _, (params, obs) in zip(trange(self.nbatches), self.dataset):
            post = self.nrep.post(obs, head, tail)
            for key, val in self.nrep.perc(params, post).items():
                qs[key] += val.tolist()
            for key, val in post.items():
                norms[key] += (val.mean(key)).tolist()

        return qs, pd.DataFrame(norms).mul(pd.Series({g: plotter.range_area for g, plotter in self.nrep.plotters.items()}), axis='columns')

    def plot_qq(self, qs, fig=None) -> plt.Figure:
        if fig is None:
            fig = plt.figure()
        ax = self.nrep.qq(qs, sigmas=False)
        ax.legend()
        ax.set_aspect('equal')
        return fig

    def plot_norms(self, norms, fig=None) -> plt.Figure:
        if fig is None:
            fig = plt.figure()

        plt.axhline(1., color='k')
        sns.violinplot(data=pd.DataFrame({
            self.nrep.group_label(g): col for g, col in norms.items()
        }), cut=0., inner='box', saturation=1., scale='width')
        plt.suptitle('NRE posterior normalisation')
        plt.ylim(0.5, 1.5)

        return fig

    def __call__(self, head: _HeadT, tail: _TailT):
        qs, norms = self.simulate(head, tail)
        return self.plot_qq(qs), self.plot_norms(norms)
