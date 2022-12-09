from __future__ import annotations

from typing import Sequence

import pandas as pd
from matplotlib import pyplot as plt
from more_itertools import always_iterable


class LossPlotter:
    def __init__(self, losses: Sequence[float], smoothing=100):
        self.losses = pd.Series(losses, name='loss')
        self.slosses = self.losses.rolling(smoothing, center=True, min_periods=0) if smoothing else self.losses

    @staticmethod
    def _prep_ax(ax: plt.Axes = None):
        if ax is None:
            ax = plt.gca()
            ax.set_xlabel('step')
            ax.grid()
        return ax

    def plot(self, quantiles=(0.1, 0.9), ax: plt.Axes = None):
        ax = self._prep_ax(ax)
        ax.set_ylabel('loss')

        ax.plot(self.slosses.mean())
        for q in always_iterable(quantiles):
            ax.plot(self.slosses.quantile(q))
        return ax

    def plot_relative(self, ax: plt.Axes = None):
        m = self.losses.min()
        ax = self._prep_ax(ax)
        ax.set_ylabel(f'loss - {float(m):.0f}')

        ax.semilogy(self.slosses.mean() - m + 1)
        return ax
