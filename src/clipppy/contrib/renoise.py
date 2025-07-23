from dataclasses import dataclass
from numbers import Number
from typing import Iterable, MutableMapping

import torch
from torch import Tensor
from torch.utils.data.datapipes.iter import IterableWrapper

from ..utils.typing import _KT, _VT


@dataclass
class GaussianRenoiser:
    noise: Tensor | Number
    noiseless_name: _KT
    noisy_name: _KT

    def __call__(self, item):
        nsless = item[self.noiseless_name]
        if torch.is_tensor(self.noise):
            self.noise = self.noise.to(dtype=nsless.dtype, device=nsless.device)
        item[self.noisy_name] = nsless + self.noise * torch.randn_like(nsless)
        return item


def GaussianRenoise(dataset: Iterable[MutableMapping[_KT, _VT]], noise: Tensor | Number, noiseless_name: _KT, noisy_name: _KT):
    return IterableWrapper(dataset).map(GaussianRenoiser(noise, noiseless_name, noisy_name))
