from __future__ import annotations

from itertools import count

from emcee import EnsembleSampler

from .mcmc import NumpyMCMCHelper


class Emcee(NumpyMCMCHelper, EnsembleSampler):
    def __init__(self, model, nwalkers, exclude=(), transform=True, vectorize=True, moves=None):
        NumpyMCMCHelper.__init__(self, model, exclude, transform)
        EnsembleSampler.__init__(
            self, nwalkers, self.ndim, self.log_prob,
            vectorize=vectorize, moves=moves
        )

    def to_dataset(self):
        from xarray import Dataset

        return Dataset({
            f'{key}_{k}' if k else key: (('chain', 'draw'), v.cpu().numpy())
            for key, val in self.constrain(self.get_batched_params(self.chain)).items()
            for k, v in (zip(map(str, count()), val.flatten(2).unbind(-1)) if val.ndim > 2 else [('', val)])
        }, coords={
            'lnp': (('chain', 'draw'), self.lnprobability)
        })
