from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import numpy as np
import pyro
from emcee import EnsembleSampler
from torch import Tensor
from torch.distributions import biject_to, Transform

from .. import clipppy
from ..utils import noop, to_tensor


class Emcee(EnsembleSampler):
    def __init__(self, config: clipppy.Clipppy, nwalkers, exclude=(), vectorize=True,
                 callback=noop):
        self.config = config
        self.callback = callback

        self.param_sites = OrderedDict(
            (key, site) for key, site in
            pyro.poutine.trace(config.model).get_trace().iter_stochastic_nodes()
            if key not in exclude
        )

        self.transforms: Mapping[str, Transform] = {
            name: biject_to(site['fn'].support)
            for name, site in self.param_sites.items()
        }

        super().__init__(
            nwalkers, len(self.param_sites), self.log_prob,
            vectorize=vectorize
        )

    def __getstate__(self):
        state = super().__getstate__()
        state['config'] = None
        return state

    def constrain(self, values: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        return {key: self.transforms[key](val) for key, val in values.items()}

    def deconstrain(self, values: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        return {key: self.transforms[key].inv(val) for key, val in values.items()}

    def get_batched_params(self, params: np.ndarray) -> Mapping[str, Tensor]:
        return dict(zip(self.param_sites.keys(), to_tensor(params).unbind(-1)))

    def log_prob(self, _unconstrained_params: np.ndarray):
        unconstrained_params = self.get_batched_params(_unconstrained_params)
        params = self.constrain(unconstrained_params)

        with pyro.condition(data=params), pyro.plate('walkers', len(_unconstrained_params)):
            trace = self.config.mock(conditioning=True, initting=False)
        trace.compute_log_prob()

        self.callback(trace)

        log_prob = sum(
            t.log_abs_det_jacobian(unconstrained_params[key], params[key])
            for key, t in self.transforms.items()
        ) + sum(
            site['log_prob']
            for site in trace.nodes.values()
            if site['type'] == 'sample')
        return log_prob.detach().cpu().numpy()

    def to_dataset(self):
        from xarray import Dataset

        return Dataset({
            key: (('chain', 'draw'), val)
            for key, val in self.constrain(self.get_batched_params(self.chain)).items()
        }, coords={
            'lnp': (('chain', 'draw'), self.lnprobability)
        })
