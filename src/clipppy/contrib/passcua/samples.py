from numbers import Real
from typing import Any, NamedTuple

import torch
from torch import Tensor
from torchmetrics import SumMetric, Metric, CatMetric


class SquareSumMetric(SumMetric):
    def update(self, value):
        return super().update(value**2)


class SumSquaredMetric(SumMetric):
    def compute(self) -> Tensor:
        return super().compute().square()


class WeightsMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = CatMetric()
        self.wsum2 = SumSquaredMetric()
        self.w2sum = SquareSumMetric()

    def update(self, w: float | Tensor):
        for m in (self.w, self.wsum2, self.w2sum):
            m.update(w)

    def compute(self) -> Any:
        return self.w.compute()

    def ess(self):
        return self.wsum2.compute() / self.w2sum.compute()

    def sample(self, n: int, replacement=True, **kwargs):
        return torch.multinomial(self.w.compute(), n, replacement=replacement, **kwargs)


class SampleBatch(NamedTuple):
    p: Tensor
    y: Tensor
    w: Tensor


class SamplesMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._p = CatMetric()
        self._y = CatMetric()
        self._w = WeightsMetric()

    p: Tensor
    y: Tensor
    w: Tensor

    def __getattribute__(self, item):
        if item in ('p', 'y', 'w'):
            return getattr(self, '_'+item).compute()
        return super().__getattribute__(item)

    def update(self, **kwargs):
        for key, val in kwargs.items():
            getattr(self, '_'+key).update(val)

    def ess(self) -> Real:
        return self._w.ess().item()

    def sample(self, n: int, return_y=False, replacement=True, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        p = self.p[idx := self._w.sample(n, replacement, **kwargs)]
        return (p, self.y[idx]) if return_y else p

    def compute(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.p)

    def __repr__(self):
        return f'{type(self).__name__}(n={len(self)}, ess={self.ess()})'
