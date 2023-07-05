from typing import Sequence, Iterable

import torch
from pyro.distributions import TorchDistribution
from torch import Tensor, Size, LongTensor

from phytorchx import broadcast_gather, mid_many


class HistDist(TorchDistribution):
    arg_constraints = {}

    def __init__(self, grids: Sequence[Tensor], log_probs: Tensor, validate_args=None):
        self.ndim = len(grids)

        assert log_probs.shape[-self.ndim:] == Size(len(g)-1 for g in grids)

        super().__init__(batch_shape=torch.broadcast_shapes(
            *(g.shape[:-1] for g in grids), log_probs.shape[:-self.ndim]
        ), event_shape=Size((self.ndim,)), validate_args=validate_args)

        grid_low, grid_high = (
            torch.stack(torch.broadcast_tensors(*(
                g[..., s].reshape(*g.shape[:-1], *(
                    -1 if i==j else 1 for j in range(self.ndim)
                )) for i, g in enumerate(grids)
            )), -1)
            for s in (slice(None, -1), slice(1, None))
        )

        self.log_mass = (log_probs + (grid_high - grid_low).log().sum(-1)).flatten(-self.ndim)

        self._categorical = torch.distributions.Categorical(logits=self.log_mass).expand(self.batch_shape)
        self._grids = (g.flatten(-self.ndim-1, -2) for g in (grid_low, grid_high))

    @classmethod
    def from_samples(cls, samples: Tensor, bins, range=None, weight=None, validate_args=None):
        # TODO: histogramdd on CUDA
        hist, bin_edges = torch.histogramdd(samples.cpu(), bins, range=range, weight=weight.cpu(), density=True)
        return cls([e.to(samples.device) for e in bin_edges], hist.to(samples.device).log(), validate_args=validate_args)

    has_rsample = True

    @property
    def _grids(self) -> tuple[Tensor, Tensor]:
        return self._grid_low, self._grid_high

    @_grids.setter
    def _grids(self, value: Iterable[Tensor]):
        self._grid_low, self._grid_high = value

    def rsample(self, sample_shape=Size()):
        idx = self._categorical.sample(sample_shape)
        return torch.lerp(*(
            broadcast_gather(g, -2, idx, index_ndim=0)
            for g in self._grids
        ), torch.rand(idx.shape + (self.ndim,)))

    def poisson_sample(self, sample_shape=Size()) -> tuple[Tensor, LongTensor]:
        counts = torch.distributions.Poisson(self.log_mass.exp()).sample(sample_shape).to(int)
        gshape = counts.shape + (self.ndim,)

        return (
            torch.lerp(*(torch.repeat_interleave(
                g.expand(gshape).flatten(end_dim=-2),
                counts.flatten(), dim=-2,
                output_size=counts.sum()
            ) for g in self._grids), torch.rand(counts.sum(), 1)),
            counts.sum(-1)
        )


class NDIDistribution2(HistDist):
    def __init__(self, grids: Sequence[Tensor], log_probs: Tensor, validate_args=None):
        super().__init__(grids, mid_many(log_probs, range(-len(grids), 0)), validate_args=validate_args)