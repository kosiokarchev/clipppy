from dataclasses import dataclass
from typing import Iterable, TypeVar, MutableMapping

import torch
from torch import Tensor


_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


@dataclass
class GaussianRenoise:
    dataset: Iterable[MutableMapping[_KT, _VT]]
    noise: Tensor
    noiseless_name: _KT
    noisy_name: _KT

    def __iter__(self):
        for item in self.dataset:
            nsless = item[self.noiseless_name]
            self.noise = torch.as_tensor(self.noise, dtype=nsless.dtype, device=nsless.device)
            item[self.noisy_name] = nsless + self.noise * torch.randn_like(nsless)
            yield item
