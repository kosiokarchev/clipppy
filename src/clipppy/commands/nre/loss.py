from __future__ import annotations

import collections.abc
from dataclasses import dataclass
from functools import reduce, singledispatchmethod
from itertools import starmap
from operator import add
from typing import Callable, Generic, Iterable, TypeVar, Union

from torch import Tensor
from torch.nn.functional import logsigmoid

from ...utils import Sentinel


__all__ = 'NRELoss',


_T = TypeVar('_T')


@dataclass
class CountReducer(Generic[_T]):
    func: Callable[[_T, _T], _T]
    n: int = 1

    def __call__(self, a: _T, b: _T) -> _T:
        self.n += 1
        return self.func(a, b)


def mean(iterable: Iterable[_T]) -> _T:
    return reduce(_ := CountReducer(add), iterable) / _.n


@dataclass
class NRELoss:
    dim: Union[int, tuple[int, ...], tuple[str, ...]] = Sentinel.skip

    @singledispatchmethod
    def __call__(self, *args, **kwargs) -> Tensor:
        raise TypeError

    @__call__.register
    def _call_tensor(self, log_ratio_joint: Tensor, log_ratio_marginal: Tensor) -> Tensor:
        return - (logsigmoid(log_ratio_joint) + logsigmoid(-log_ratio_marginal)).mean(*(
            (self.dim,) if self.dim is not Sentinel.skip else ()
        ))

    @__call__.register(collections.abc.Iterable)
    def _call_iterable(self, log_ratio_joint: Iterable[Tensor], log_ratio_marginal: Iterable[Tensor]) -> Tensor:
        return mean(starmap(self._call_tensor, zip(log_ratio_joint, log_ratio_marginal)))
