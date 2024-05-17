from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar, Any, Generic, Mapping

import numpy as np
import pyro
import torch

from torch import Tensor
from torch.distributions import Transform, biject_to

from clipppy.utils import to_tensor
from clipppy.utils.trace import ClipppyTraceMessenger

_ArrayT = TypeVar('_ArrayT', Tensor, Any)


class MCMCHelper(Generic[_ArrayT]):
    def __init__(self, model, exclude=(), transform=True):
        self.model = model
        self._transform = transform

        self.sites = OrderedDict(
            (key, site) for key, site in
            pyro.poutine.trace(model).get_trace().iter_stochastic_nodes()
            if key not in exclude
        )
        self.ndim = sum(s['value'].numel() for s in self.sites.values())

        self.transforms: Mapping[str, Transform] = OrderedDict(
            (name, biject_to(site['fn'].support))
            for name, site in self.sites.items()
        )

    def __getstate__(self):
        state = super().__getstate__()
        state['model'] = None
        return state

    def constrain(self, values: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        return {key: self.transforms[key](val) for key, val in values.items()} if self._transform else values

    def deconstrain(self, values: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        return {key: self.transforms[key].inv(val) for key, val in values.items()} if self._transform else values

    def get_batched_params(self, params: _ArrayT) -> Mapping[str, Tensor]:
        return {
            key: params[..., i-s:i].reshape((*params.shape[:-1], *val.shape))
            for i in [0]
            for key, site in self.sites.items() for val in [site['value']]
            for s in [val.numel()] for i in [i+s]
        }

    def get_param_vector(self, params: Mapping[str, Tensor]) -> _ArrayT:
        return torch.cat([
            val.flatten(val.ndim-site['value'].ndim)
            for key, site in self.sites.items()
            for val in [params[key]]
        ], dim=-1)

    def pack(self, init: Mapping[str, Tensor]) -> _ArrayT:
        return self.get_param_vector(self.deconstrain({key: init[key] for key in self.sites.keys()}))

    def log_prob(self, _unconstrained_params: _ArrayT) -> _ArrayT:
        unconstrained_params = self.get_batched_params(_unconstrained_params)
        params = self.constrain(unconstrained_params)

        with (
            ClipppyTraceMessenger() as tracer,
            pyro.condition(data=params),
            pyro.plate_stack('batch', _unconstrained_params.shape[:-1])
        ):
            self.model()

        log_prob = tracer.get_trace().log_prob()
        if self._transform:
            log_prob = log_prob + sum(
                t.log_abs_det_jacobian(unconstrained_params[key], params[key])
                for key, t in self.transforms.items()
            )
        return log_prob


class NumpyMCMCHelper(MCMCHelper[np.ndarray]):
    def get_batched_params(self, params: np.ndarray) -> Mapping[str, Tensor]:
        return super().get_batched_params(to_tensor(params))

    def get_param_vector(self, params: Mapping[str, Tensor]) -> np.ndarray:
        return super().get_param_vector(params).numpy(force=True)

    def log_prob(self, _unconstrained_params: np.ndarray) -> np.ndarray:
        return super().log_prob(_unconstrained_params).numpy(force=True)
