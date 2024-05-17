from typing import Callable

import numpy as np
import torch
from dynesty import NestedSampler
from torch import Tensor

from ..contrib.minspect import PackedSites
from ..utils.messengers import SimpleReplayMessenger, UnitCubePrior
from ..utils.trace import ClipppyTraceMessenger


def numpy_inout(fn: Callable[[Tensor], Tensor], dtype: torch.dtype = None, device: torch.device = None) -> Callable[[np.ndarray], np.ndarray]:
    # TODO: update wrapper but allow pickling?
    def _fn(arg: np.ndarray) -> np.ndarray:
        return fn(torch.tensor(arg, dtype=dtype, device=device)).numpy(force=True)
    return _fn


class Dynesty:
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.ps = PackedSites.from_model(model, *args, **kwargs)
        self.ndim = self.ps.event_shape.numel()

    def sampler(self, dtype=None, device=None, **kwargs) -> NestedSampler:
        return NestedSampler(self.sampler_args(dtype, device), **kwargs)

    def sampler_args(self, dtype=None, device=None):
        return (
            numpy_inout(self.loglikelihood, dtype=dtype, device=device),
            numpy_inout(self.prior_transform, dtype=dtype, device=device),
            self.ndim
        )

    def prior_transform(self, u: Tensor) -> Tensor:
        with (
            UnitCubePrior(*self.ps.sites.keys()) as ucp,
            SimpleReplayMessenger({ucp._ucp_name(key): val for key, val in self.ps.unpack(u).items()})
        ):
            self.model()
        return self.ps.pack(ucp.as_valuedict())

    def loglikelihood(self, x: Tensor):
        with ClipppyTraceMessenger() as tracer, SimpleReplayMessenger(self.ps.unpack(x)):
            self.model()
        return tracer.get_trace().log_likelihood()
