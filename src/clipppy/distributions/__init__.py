from typing import Sequence

import torch
from more_itertools import one, unique_everseen
from pyro.distributions import Categorical, Rejector, TransformedDistribution
from pyro.distributions.torch_distribution import TorchDistribution
from torch import Size
from torch.distributions.constraints import interval
from torch.distributions.utils import broadcast_all

from phytorchx import broadcast_gather, broadcast_stack


class SupportedTransformedDistribution(TransformedDistribution):
    def transform(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    @property
    def support(self):
        support = self.base_dist.support
        assert isinstance(support, interval)
        return interval(*map(self.transform, broadcast_all(support.lower_bound, support.upper_bound)))


class SupportedRejector(Rejector):
    @property
    def support(self):
        return self.propose.support


class MixtureDistribution(TorchDistribution):
    arg_constraints = {}

    def __init__(self, dists: Sequence[TorchDistribution], probs=None, logits=None):
        self.cat = Categorical(probs=probs, logits=logits)
        self.dists = dists

        super().__init__(
            torch.broadcast_shapes(*(dist.batch_shape for dist in self.dists)),
            one(unique_everseen(dist.event_shape for dist in self.dists))
        )

    def sample(self, sample_shape=Size()):
        index_dim = len(sample_shape) + len(self.batch_shape)
        return broadcast_gather(
            broadcast_stack(
                (dist.sample(sample_shape) for dist in self.dists),
                index_dim
            ), dim=index_dim,
            index=self.cat.sample(sample_shape), index_ndim=0
        )

    def log_prob(self, x, *args, **kwargs):
        return (broadcast_stack((dist.log_prob(x, *args, **kwargs) for dist in self.dists), -1) + self.cat.logits).logsumexp(-1)
