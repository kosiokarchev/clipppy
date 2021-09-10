from torch.distributions import ComposeTransform

from .guide import Guide
from .group_spec import GroupSpec, GroupSpec
from .sampling_groups import (
    DeltaSamplingGroup, DiagonalNormalSamplingGroup,
    MultivariateNormalSamplingGroup, PartialMultivariateNormalSamplingGroup
)


def fix_transform(t):
    if isinstance(t, ComposeTransform):
        for p in t.parts:
            fix_transform(p)
        return
    if not hasattr(t, '_event_dim'):
        t._event_dim = 0
    print('fixed', t)


def fix_guide(g: Guide):
    for group in g.children():
        for t in group.transforms.values():
            fix_transform(t)
            fix_transform(t.inv)
    return g


__all__ = ('DeltaSamplingGroup', 'DiagonalNormalSamplingGroup', 'MultivariateNormalSamplingGroup',
           'PartialMultivariateNormalSamplingGroup', 'GroupSpec', 'Guide')
