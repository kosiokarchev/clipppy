from functools import partial
from typing import Mapping, Type, Callable, Iterable, TypeVar, Generic

import torch
from pyro.distributions import Independent
from torch import LongTensor, Tensor
from torch.distributions import Normal, Uniform
from typing_extensions import Self

from .extra_dimensions import ExtraBatched
from .wrapper import DistributionWrapper, _Distribution

_CKT = TypeVar('_CKT')
_VT = TypeVar('_VT')


class ClassKeyDict(dict[Type[_CKT], _VT], Generic[_CKT, _VT]):
    def __getitem__(self, item):
        if item not in self:
            for key in self.keys():
                if issubclass(item, key):
                    self[item] = super().__getitem__(key)
                    break
        return super().__getitem__(item)


def ubexpand(t: Tensor, sizes: Tensor, dim=-1):
    return torch.repeat_interleave(
        t.expand(sizes.shape + t.shape[(t.ndim and dim % t.ndim) + 1:]).flatten(end_dim=dim),
        sizes.flatten(), dim=0,
        output_size=sizes.sum()
    )


class UBSDistribution(DistributionWrapper):
    """Un(even|equal)-Batch Sampling Distribution"""

    _param_map: Mapping[Type, Callable[[_Distribution], Iterable[tuple[Tensor, int]]]] = ClassKeyDict({
        Normal: lambda d: ((d.loc, 0), (d.scale, 0)),
        Uniform: lambda d: ((d.low, 0), (d.high, 0)),
    })

    @classmethod
    def ubexpand(cls, base_dist: _Distribution, sizes: LongTensor) -> Self:
        wrapper, bd = lambda d: d, base_dist
        if isinstance(base_dist, Independent):
            wrapper = partial(Independent, reinterpreted_batch_ndims=base_dist.reinterpreted_batch_ndims)
            bd = base_dist.base_dist
        base_type = type(bd)

        params = cls._param_map[base_type](base_dist)
        batch_shape = torch.broadcast_shapes(base_dist.batch_shape, sizes.shape)

        # if isinstance(base_dist, ConUnDisMixin):
        #     params += (base_dist.constraint_lower, base_dist.constraint_upper)

        return cls(wrapper(base_type(*(
            ubexpand(param, torch.broadcast_to(sizes, batch_shape), -1-event_dims)
            for param, event_dims in params
        ))), batch_shape=batch_shape, event_shape=base_dist.event_shape)

    def expand(self, batch_shape, _instance=None):
        return ExtraBatched(self, batch_shape[:-self.batch_dim])
