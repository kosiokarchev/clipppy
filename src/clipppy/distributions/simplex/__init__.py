from __future__ import annotations

from functools import partial
from typing import Iterable

import torch
from pyro.distributions import TorchDistribution
from torch import Tensor
from torch.distributions.constraints import Constraint
from typing_extensions import Self

from .utils import points_to_simplices, points_to_simplices_voronoi

try:
    from matplotlib.path import Path as mplPath
except ImportError:
    mplPath = None


class SimplexDistribution(TorchDistribution):
    simplices: Tensor
    """``(batch_shape..., nsimplices, ndim+1, ndim)``"""

    weights: Tensor
    """``(batch_shape..., nsimplices)``"""

    arg_constraints = {}
    has_rsample = True

    def __init__(self, simplices: Tensor, weights: Tensor):
        assert simplices.shape[-3] == weights.shape[-1]
        super().__init__(
            torch.broadcast_shapes(simplices.shape[:-3], weights.shape[:-1]),
            simplices.shape[-1:]
        )

        self.simplices = simplices.expand(self.batch_shape + simplices.shape[-3:])
        self.weights = (weights / weights.sum(-1, keepdim=True)).expand(self.batch_shape + weights.shape[-1:])

    @classmethod
    def from_samples(cls, pts: Tensor, pweights: Tensor = None, voronoi=False, constraint: Constraint = None, **kwargs) -> Self:
        if pweights is None:
            pweights = pts.new_ones(pts.shape[-2])

        simplices, sweights = (points_to_simplices_voronoi if voronoi else points_to_simplices)(pts, pweights, **kwargs)

        if constraint:
            valid = constraint.check(simplices).all(-1)
            simplices, sweights = simplices[valid], sweights[valid]

        return cls(simplices, sweights)

    @classmethod
    def from_contour(cls, paths: Iterable[mplPath], device=None, dtype=None):
        from .contours import contour_to_simplices
        return cls(*contour_to_simplices(paths, device, dtype))

    def sample_simplices(self, sample_shape=torch.Size()):
        return (partial(Tensor.unflatten, sizes=sample_shape) if sample_shape else Tensor.squeeze)(
            self.simplices.take_along_dim(
                torch.distributions.Categorical(probs=self.weights)
                .sample(sample_shape)
                .flatten(end_dim=len(sample_shape)-1)
                .movedim(0, -1)[..., None, None],
                dim=-3
            ).movedim(-3, 0),
            0
        )

    @staticmethod
    def sample_within_simplices(simplices: Tensor):
        return torch.linalg.vecdot(
            torch.distributions.Dirichlet(simplices.new_ones(simplices.shape[:-1])).sample().unsqueeze(-1),
            simplices, dim=-2
        )

    def rsample(self, sample_shape=torch.Size()):
        return self.sample_within_simplices(self.sample_simplices(sample_shape))
