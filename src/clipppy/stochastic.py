import abc
import types
import typing as tp
from contextlib import nullcontext
from functools import partial, update_wrapper, wraps

import pyro
import torch
from pyro import distributions as dist
from pyro.contrib.autoname import scope

__all__ = ('Sampler', 'Param', 'InfiniteSampler', 'SemiInfiniteSampler',
           'StochasticWrapper', 'stochastic')

_sentinel = object()


class InfiniteUniform(dist.TorchDistribution):
    @property
    def arg_constraints(self):
        return {}

    def rsample(self, sample_shape=torch.Size()):
        return torch.zeros(sample_shape)

    @dist.constraints.dependent_property
    def support(self):
        return dist.constraints.real

    def log_prob(self, value):
        return torch.zeros_like(value)


class SemiInfiniteUniform(InfiniteUniform):
    @dist.constraints.dependent_property
    def support(self):
        return dist.constraints.positive

    def log_prob(self, value):
        return torch.where(value < 0., value.new_full((), -float('inf')), value.new_zeros(()))


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


InfiniteSampler = wraps(Sampler)(partial(Sampler, d=InfiniteUniform()))
SemiInfiniteSampler = wraps(Sampler)(partial(Sampler, d=SemiInfiniteUniform()))


class StochasticWrapper:
    stochastic_name: str
    stochastic_specs: tp.Mapping[str, tp.Union[Sampler, tp.Any]]

    def __call__(self, *args, **kwargs):
        with scope(prefix=self.stochastic_name) if self.stochastic_name else nullcontext():
            return super(type(self), self).__call__(*args, **kwargs, **{
                name: (spec() if isinstance(spec, AbstractSampler) else spec)
                for name, spec in self.stochastic_specs.items()
                if name not in kwargs
            })

    def __repr__(self):
        return (f'"{self.stochastic_name}": ' if self.stochastic_name else '') + super(type(self), self).__repr__()


class ObjectWrapper:
    def __init__(self, baseObject):
        self.__class__ = baseObject.__class__
        self.__dict__ = baseObject.__dict__


class FunctionWrapper:
    """def functions and lambdas in python are special..."""
    def __init__(self, func: types.FunctionType):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def _stochastic(obj, stochastic_specs: tp.Mapping[str, tp.Union[Sampler, tp.Any]], name=None):
    """Make a StochasticWrapper from an object, with the given specs and name"""
    _obj = FunctionWrapper(obj) if isinstance(obj, types.FunctionType) else ObjectWrapper(obj)

    _obj.__class__ = type(f'StochasticWrapper<{type(_obj).__name__}>', (type(_obj),), {
        '__annotations__': StochasticWrapper.__annotations__,
        '__call__': update_wrapper(StochasticWrapper.__call__, type(_obj).__call__),
        '__reduce__': lambda self: (_stochastic, (obj, stochastic_specs, name)),
        '__repr__': StochasticWrapper.__repr__
    })
    _obj.stochastic_specs = stochastic_specs
    _obj.stochastic_name = name

    return _obj


def stochastic(obj, specs: tp.Mapping[str, tp.Union[Sampler, dist.torch_distribution.TorchDistributionMixin, tp.Any]], name=None):
    """
    Make a StochasticWrapper from a dict of specs that can be:
       - full-blown Sampler's
       - TorchDistributionMixin's that will be wrapped in Sampler's
       - any other value that will be passed as-is
    If you really want to pass a distribution to the wrapper as-is,
    use :any:`_stochastic`.
    """
    return _stochastic(obj=obj, stochastic_specs={
        name: (spec.set_name(name) if isinstance(spec, Sampler)
               else Sampler(spec, name=name) if isinstance(spec, dist.torch_distribution.TorchDistributionMixin)
               else spec)
        for name, spec in specs.items()
    }, name=name)


from .globals import register_globals
register_globals(**{a: globals()[a] for a in __all__ if a in globals()})
