from __future__ import annotations

# noinspection PyUnresolvedReferences
from typing import TYPE_CHECKING

from pyro.distributions import Distribution, TransformedDistribution


if TYPE_CHECKING:
    import torch
    from typing import Type, Union

    # because TransformedDistribution cannot be found in pyro.distributions
    TransformedDistribution: Union[Type[torch.distributions.TransformedDistribution], Type[Distribution]]
