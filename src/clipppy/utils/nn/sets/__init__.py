from __future__ import annotations

from functools import partial
from itertools import cycle
from typing import Callable, Iterable, Literal, Union

import attr
from more_itertools import last, always_iterable
from torch import Tensor
from torch.nn import Module, ModuleList
from torch_scatter import segment_csr
from typing_extensions import TypeAlias

from ..attrs import AttrsModule


_collapse_fn_t: TypeAlias = Callable[[Tensor, Iterable[int], int], Tensor]


@attr.s(auto_attribs=True, eq=False)
class NestedSetsProcessor(AttrsModule):
    @staticmethod
    def _collapse(t: Tensor, lens: Iterable[int], dim: int, reduce: Literal['mean', 'sum']):
        return segment_csr(
            t,
            t.new_tensor((
                0, *(i for i in [0] for l in lens for i in [i+l])
            ), dtype=int).view(
                (*(1 for _ in range(dim % t.ndim)), -1)
            ),
            reduce=reduce
        )

    collapse_sum = staticmethod(partial(_collapse.__func__, reduce='sum'))
    collapse_mean = staticmethod(partial(_collapse.__func__, reduce='mean'))

    nets: Iterable[Module]
    collapse_fns: Union[_collapse_fn_t, Iterable[_collapse_fn_t]] = collapse_mean.__func__

    def __attrs_post_init__(self):
        if not isinstance(self.nets, ModuleList):
            self.nets = ModuleList(self.nets)

    def forward(self, t: Tensor, lenss: Iterable[Iterable[int]], dim=-2):
        return last(
            t for t in [t]
            for net, lens, collapse_fn in zip(self.nets, lenss, cycle(always_iterable(self.collapse_fns)))
            for t in [collapse_fn(net(t), lens, dim)]
        )
