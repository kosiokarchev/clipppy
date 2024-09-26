from dataclasses import dataclass
from numbers import Number
from typing import Iterable, TypeVar, MutableMapping, Union

import torch
from torch import Tensor
from torchdata.datapipes.iter import IterableWrapper


_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


@dataclass
class GaussianRenoiser:
    noise: Union[Tensor, Number]
    noiseless_name: _KT
    noisy_name: _KT

    def __call__(self, item):
        nsless = item[self.noiseless_name]
        if torch.is_tensor(self.noise):
            self.noise = self.noise.to(dtype=nsless.dtype, device=nsless.device)
        item[self.noisy_name] = nsless + self.noise * torch.randn_like(nsless)
        return item


def GaussianRenoise(dataset: Iterable[MutableMapping[_KT, _VT]], noise: Union[Tensor, Number], noiseless_name: _KT, noisy_name: _KT):
    return IterableWrapper(dataset).map(GaussianRenoiser(noise, noiseless_name, noisy_name))
