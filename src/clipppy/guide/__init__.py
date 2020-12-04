from .group_spec import GroupSpec
from .guide import Guide
from .sampling_groups import (
    DeltaSamplingGroup, DiagonalNormalSamplingGroup,
    MultivariateNormalSamplingGroup, PartialMultivariateNormalSamplingGroup
)
from ..globals import register_globals

__all__ = ('DeltaSamplingGroup', 'DiagonalNormalSamplingGroup', 'MultivariateNormalSamplingGroup',
           'PartialMultivariateNormalSamplingGroup', 'GroupSpec', 'Guide')

register_globals(**{a: globals()[a] for a in __all__ if a in globals()})
