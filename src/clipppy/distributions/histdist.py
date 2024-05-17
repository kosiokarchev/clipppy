from typing import Sequence, Iterable, cast, Optional, Any

import torch
from pyro.distributions import TorchDistribution
from torch import Tensor, Size, LongTensor, BoolTensor
from torch.distributions import constraints
from torch.distributions.constraints import interval

from phytorch.interpolate import Linear1dInterpolator
from phytorchx import broadcast_gather, broadcast_multigather, mid_many, mid_one


class HistDist0d(TorchDistribution):
    arg_constraints = {}

    def __init__(self, grid_edges: Tensor, log_prob: Tensor, **kwargs):
        super().__init__(batch_shape=torch.broadcast_shapes(grid_edges.shape[:-1], log_prob.shape[:-1]), **kwargs)
        self.grid_edges = grid_edges
        self.log_mass = log_prob + grid_edges.diff(dim=-1).log()
        self.log_total_mass = self.log_mass.logsumexp(-1)
        self.grid_log_prob = log_prob - self.log_total_mass.unsqueeze(-1)

        grid_cdf = self.log_mass.logcumsumexp(-1)
        grid_cdf = torch.cat((
            grid_cdf.new_zeros(()).expand(*grid_cdf.shape[:-1], 1),
            (grid_cdf - grid_cdf[..., -1:]).exp()
        ), dim=-1)

        self.log_prob_interp = Linear1dInterpolator(mid_one(self.grid_edges, -1), self.grid_log_prob, channel_ndim=0)
        self.cdf_interp = Linear1dInterpolator(self.grid_edges, grid_cdf, channel_ndim=0)
        self.icdf_interp = Linear1dInterpolator(grid_cdf, self.grid_edges, channel_ndim=0)

    @constraints.dependent_property
    def support(self):
        return interval(*self.grid_edges[..., (0, -1)].unbind(-1))

    # def log_prob(self, value):
    #     return broadcast_gather(
    #         self.grid_log_prob, -1,
    #         cast(LongTensor, torch.searchsorted(self.grid_edges, value)-1),
    #         index_ndim=0
    #     )

    def log_prob(self, value):
        return self.log_prob_interp(value)

    def cdf(self, value):
        return self.cdf_interp(value)

    def icdf(self, value):
        return self.icdf_interp(value)


class HistDist(TorchDistribution):
    arg_constraints = {}

    def __init__(self, grid_edges: Sequence[Tensor], log_probs: Tensor, validate_args=None):
        self.ndim = len(grid_edges)

        assert log_probs.shape[-self.ndim:] == Size(len(g) - 1 for g in grid_edges)

        super().__init__(batch_shape=torch.broadcast_shapes(
            *(g.shape[:-1] for g in grid_edges), log_probs.shape[:-self.ndim]
        ), event_shape=Size((self.ndim,)), validate_args=validate_args)

        self.grid_edges = grid_edges
        self.grid_bounds = *torch.stack([g[(0, -1),] for g in grid_edges], -1),

        grid_low, grid_high = (
            torch.stack(torch.broadcast_tensors(*(
                g[..., s].reshape(*g.shape[:-1], *(
                    -1 if i==j else 1 for j in range(self.ndim)
                )) for i, g in enumerate(grid_edges)
            )), -1)
            for s in (slice(None, -1), slice(1, None))
        )

        self.grid_log_prob = log_probs - log_probs.logsumexp(tuple(range(-self.ndim, 0)), keepdim=True)
        self.log_mass = (log_probs + (grid_high - grid_low).log().sum(-1)).flatten(-self.ndim)

        self._categorical = torch.distributions.Categorical(logits=self.log_mass).expand(self.batch_shape)
        self._grids = (g.flatten(-self.ndim-1, -2) for g in (grid_low, grid_high))

    @classmethod
    def from_samples(cls, samples: Tensor, bins, range=None, weight=None, validate_args=None):
        # TODO: histogramdd on CUDA
        hist, bin_edges = torch.histogramdd(samples.cpu(), bins, range=range, weight=weight.cpu(), density=True)
        return cls([e.to(samples.device) for e in bin_edges], hist.to(samples.device).log(), validate_args=validate_args)

    @property
    def _grids(self) -> tuple[Tensor, Tensor]:
        return self._grid_low, self._grid_high

    @_grids.setter
    def _grids(self, value: Iterable[Tensor]):
        self._grid_low, self._grid_high = value

    @constraints.dependent_property
    def support(self):
        return constraints.independent(interval(self.grid_bounds[0], self.grid_bounds[1]), 1)

    def in_bounds(self, value: Tensor) -> BoolTensor:
        return torch.logical_and(
            value >= self.grid_bounds[0],
            value <= self.grid_bounds[1]
        ).all(-1)

    def log_prob(self, value):
        return broadcast_multigather(self.grid_log_prob, *(
            torch.searchsorted(g, v).clamp(0, len(g)-2)
            for g, v in zip(self.grid_edges, value.unbind(-1))
        )).where(self.in_bounds(value), -float('inf'))

    has_rsample = True

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
    def __init__(self, grid_edges: Sequence[Tensor], log_probs: Tensor, validate_args=None):
        super().__init__(grid_edges, mid_many(log_probs, range(-len(grid_edges), 0)), validate_args=validate_args)
