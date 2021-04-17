import abc
from typing import Iterable, Union

import pyro
import torch
from pyro import distributions as dist


__all__ = 'Sampler', 'Param'


_sentinel = object()


class AbstractSampler(abc.ABC):
    def __init__(self, name: str = None, init: torch.Tensor = None, to_event: int = None,
                 support: dist.constraints.Constraint = _sentinel):
        self.name = name
        self.init = init
        self.support = support
        self.event_dim = to_event

    def set_name(self, name):
        self.name = name
        return self

    def __call__(self):
        raise NotImplemented


class Sampler(AbstractSampler):
    def __init__(self, d: dist.torch_distribution.TorchDistributionMixin,
                 expand_by: tp.Union[torch.Size, tp.Iterable[int]] = torch.Size(), mask: torch.Tensor = None,
                 name: str = None, init: torch.Tensor = None, to_event: int = None,
                 support: dist.constraints.Constraint = _sentinel,
                 **kwargs):
        super(Sampler, self).__init__(name=name, init=init, to_event=to_event, support=support)
        self.d = d
        self.expand_by = expand_by
        self.infer = dict(init=self.init, mask=mask, support=support, **kwargs)

    @property
    def infer_msgr(self):
        return pyro.poutine.infer_config(config_fn=lambda site: {key: val for key, val in self.infer.items() if val is not _sentinel})

    def __call__(self):
        with self.infer_msgr:
            return pyro.sample(self.name, self.d.expand_by(self.expand_by).to_event(self.event_dim))


class Param(AbstractSampler):
    def __call__(self):
        return pyro.param(self.name, init_tensor=self.init, constraint=self.support, event_dim=self.event_dim)
