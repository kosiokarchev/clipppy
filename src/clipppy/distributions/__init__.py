from pyro.distributions import TransformedDistribution
from torch.distributions.constraints import interval
from torch.distributions.utils import broadcast_all


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
