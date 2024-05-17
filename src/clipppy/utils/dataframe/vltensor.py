from __future__ import annotations

from functools import partial
from typing import Iterable

from torch import Tensor
from torch.utils.data._utils.collate import default_collate_fn_map


class VLTensor(Tensor):
    pass


@partial(default_collate_fn_map.__setitem__, VLTensor)
def collate_vltensor(batch: Iterable[VLTensor], *args, **kwargs):
    return list(el.as_subclass(Tensor) for el in batch)
    # return torch.nested.as_nested_tensor(list(el.as_subclass(Tensor) for el in batch))
