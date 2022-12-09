from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from math import pi
from typing import Collection, Mapping, Union

import numpy as np
import pyro
from corner import corner
from matplotlib import pyplot as plt
from pyro.poutine import Trace
from torch import Tensor

from ...guide import Guide, HPMVN


@dataclass
class HPMVNPlotter:
    ppd: Union[Trace, Mapping[str, Tensor]]
    names_global: Collection[str] = None
    names_hier: Collection[str] = None
    truths: Mapping[str, Tensor] = None

    def __post_init__(self):
        all_keys = (self.ppd.nodes if (is_trace := isinstance(self.ppd, Trace)) else self.ppd).keys()

        if self.names_global is None:
            self.names_global = all_keys - set(self.names_hier)
        if self.names_hier is None:
            self.names_hier = all_keys - set(self.names_global)

        if is_trace:
            self.ppd = {key: self.ppd.nodes[key]['value'] for key in chain(self.names_global, self.names_hier)}


    @classmethod
    def from_hpmvn(cls, hpmvn: HPMVN, nsamples=1000, truths=None):
        with pyro.plate('plate', nsamples):
            return cls(hpmvn()[1], hpmvn.sites_full.keys(), hpmvn.corr_names, truths=truths)

    @classmethod
    def from_guide(cls, guide: Guide, names_global=None, names_hier=None, nsamples=1000, truths=None):
        with pyro.plate('plate', nsamples):
            return cls(guide(), names_global, names_hier, truths=truths)

    @cached_property
    def means(self):
        return {key: val.mean(0) for key, val in self.ppd.items()}

    @cached_property
    def stds(self):
        return {key: val.std(0) for key, val in self.ppd.items()}

    @cached_property
    def errs(self):
        return {key: self.means[key] - self.truths[key] for key in self.ppd.keys()}

    @cached_property
    def serrs(self):
        return {key: self.errs[key] / self.stds[key] for key in self.ppd.keys()}

    @cached_property
    def samples_global(self):
        return {key: self.ppd[key] for key in self.names_global}

    def plot_corner(self, levels=(0.68, 0.95), plot_density=False, no_fill_contours=True, **kwargs):
        return corner(self.samples_global, truths=self.truths, **locals())

    def plot_hier(self):
        fig, axs = plt.subplots(len(self.names_hier), 2, figsize=(7, 2.5*len(self.names_hier) + 1), gridspec_kw=dict(width_ratios=(1.6, 1)), squeeze=False)
        for ax, param in zip(axs, self.names_hier):
            ax[0].axhline(0)
            ax[0].errorbar(self.truths[param], self.errs[param], yerr=self.stds[param], ls='none',
                           capsize=2, capthick=1, elinewidth=1, marker='.', ms=2.6)
            ax[0].set_xlabel(f'true {param}')
            ax[0].set_ylabel(f'posterior - true {param}')

            ax[1].hist(self.serrs[param].numpy(), density=True)
            ax[1].plot(_x := np.linspace(-3, 3, 101), np.exp(-_x**2/2) / (2*pi)**0.5)
            ax[1].set_xlabel(f'std. dev. from true {param}')

        return fig, axs
