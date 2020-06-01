"""
Guide functionality mostly copied from pyro.contrib.easyguide
with little alterations of some annoying bits and cosmetic improvement.
Also, with some new functionality.
"""
import re
from abc import abstractmethod, ABCMeta
from contextlib import ExitStack
from functools import lru_cache
from operator import itemgetter

import pyro
import torch
import typing as tp

from pyro import poutine, distributions as dist
from pyro.infer.autoguide.guides import prototype_hide_fn
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.poutine import runtime
from pyro.poutine.indep_messenger import CondIndepStackFrame
from pyro.poutine.messenger import Messenger
from torch.distributions import biject_to

from .globals import get_global, _Site, register_globals, enumlstrip, init_msgr

__all__ = ('DeltaSamplingGroup', 'DiagonalNormalSamplingGroup', 'GroupSpec', 'Guide')


class _AbstractPyroModuleMeta(type(PyroModule), ABCMeta):
    pass


class SamplingGroup(PyroModule, metaclass=_AbstractPyroModuleMeta):
    def _process_prototype(self):
        common_dims = set(f.dim
                          for site in self.sites.values()
                          for f in site['cond_indep_stack']
                          if f.vectorized)
        rightmost_common_dim = max(common_dims, default=-float('inf'))

        for name, site in self.sites.items():
            transform = biject_to(site['fn'].support)
            self.transforms[name] = transform

            self.event_shapes[name] = site['fn'].event_shape
            batch_shape = site['fn'].batch_shape

            self.batch_shapes[name] = torch.Size(
                enumlstrip(batch_shape, lambda i, x: x==1 or i-len(batch_shape) in common_dims))
            if len(self.batch_shapes[name]) > -rightmost_common_dim:
                raise ValueError(
                    'grouping expects all per-site plates to be right of all common plates, '
                    f'but found a per-site plate {-len(self.batch_shapes[name])} '
                    f'on left at site {site["name"]!r}')

            shape = self.batch_shapes[name] + self.event_shapes[name]

            init = transform.inv(site['value'].__getitem__((0,)*(site['value'].ndim - len(shape))).expand(shape))

            self.inits[name] = init

            # TODO: Wait for better days to come...:
            # _ if (_:=site.get('mask', None)) is not None
            _ = site.get('mask', None)
            self.masks[name] = (_ if _ is not None
                                else getattr(site['fn'], '_mask',
                                             init.new_full([], True, dtype=torch.bool))
                                ).expand_as(init)

            self.sizes[name] = shape.numel()

    def __init__(self, sites: tp.Iterable[_Site], name='', *args, **kwargs):
        super().__init__(name)

        self.sites: tp.Mapping[str, _Site] = {site['name']: site for site in sites}
        self.event_shapes: tp.MutableMapping[str, torch.Size] = {}
        self.batch_shapes: tp.MutableMapping[str, torch.Size] = {}
        self.sizes: tp.MutableMapping[str, int] = {}
        self.transforms: tp.MutableMapping[str, dist.transforms.Transform] = {}

        self.masks: tp.MutableMapping[str, torch.Tensor] = {}
        self.inits: tp.MutableMapping[str, torch.Tensor] = {}

        self._process_prototype()

    @property
    @lru_cache(maxsize=1)
    def init(self) -> torch.Tensor:
        return torch.cat([t.flatten() for t in self.inits.values()])

    @property
    @lru_cache(maxsize=1)
    def mask(self) -> torch.BoolTensor:
        return tp.cast(torch.BoolTensor,
                       torch.cat([m.flatten() for m in self.masks.values()]))

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size([sum(self.sizes.values())])

    @abstractmethod
    def _sample(self, infer) -> torch.Tensor: ...

    # By default, include constraining transformation's Jacobian factor
    include_det_jac = True

    def unpack(self, group_z: torch.Tensor, guide: 'Guide' = None) -> tp.Dict[str, torch.Tensor]:
        model_zs = {}
        for pos, (name, fn, frames), transform in zip(
            # lazy cumsum!! python ftw!
            (s for s in [0] for x in self.sizes.values() for s in [x+s]),
            map(itemgetter('name', 'fn', 'cond_indep_stack'),
                self.sites.values()),
            self.transforms.values()
        ):
            fn: dist.TorchDistribution
            z = group_z[..., pos-self.sizes[name]:pos]
            z = z.reshape(z.shape[:-1] + fn.event_shape)

            x = transform(z)

            if self.include_det_jac:
                log_density = transform.log_abs_det_jacobian(x, z)
                log_density = log_density.sum(list(range(-(log_density.ndim - z.ndim + fn.event_dim), 0)))
            else:
                log_density = 0.

            delta = dist.Delta(x, log_density=log_density, event_dim=fn.event_dim)

            with ExitStack() as stack:
                if guide is not None:
                    for frame in frames:
                        plate = guide.plate(frame.name)
                        if plate not in runtime._PYRO_STACK:
                            stack.enter_context(plate)
                with pyro.poutine.mask(mask=self.masks[name]):
                    model_zs[name] = pyro.sample(name, delta)

        return model_zs

    def forward(self, guide: 'Guide' = None, infer: dict = None) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor]]:
        with pyro.poutine.infer_config(
                config_fn=lambda site: dict(**(infer if infer is not None else {}), is_auxiliary=True)),\
             pyro.poutine.mask(mask=self.mask):
            group_z = self._sample(infer)

        return group_z, self.unpack(group_z, guide)

    def extra_repr(self) -> str:
        return f'{len(self.sites)} sites, {self.event_shape}'


# This should be the same as EasyGuide's map_estimate,
# but slower because of (needless?) transformations, unpacking, etc.
# TODO: simplify DeltaSamplingGroup?
class DeltaSamplingGroup(SamplingGroup):
    def __init__(self, sites, name='', *args, **kwargs):
        super().__init__(sites, name)

        self.loc = PyroParam(self.init, event_dim=1)

    # Emulate EasyGuide's map_estimate implementation
    include_det_jac = False

    def _sample(self, infer) -> torch.Tensor:
        return self.loc


class DiagonalNormalSamplingGroup(SamplingGroup):
    def __init__(self, sites, name='', init_scale=1., *args, **kwargs):
        super().__init__(sites, name, *args, **kwargs)

        self.loc = PyroParam(self.init, event_dim=1)
        self.scale = PyroParam(torch.full_like(self.init, init_scale), event_dim=1,
                               constraint=dist.constraints.positive)

        self.guide_z = PyroSample(type(self).prior)

    def prior(self):
        return dist.Normal(self.loc, self.scale).to_event(1)

    @pyro.nn.pyro_method
    def _sample(self, infer) -> torch.Tensor:
        return tp.cast(torch.Tensor, self.guide_z)


class GroupSpec:
    def __init__(self, cls: tp.Type[SamplingGroup] = DeltaSamplingGroup,
                 match: tp.Union[str, re.Pattern] = re.compile('.*'), name='', *args, **kwargs):
        if isinstance(cls, str):
            cls = get_global(cls, get_global(cls+'SamplingGroup', cls, globals()), globals())

        self.cls: tp.Type[SamplingGroup] = cls
        assert issubclass(self.cls, SamplingGroup)

        self.match = re.compile(match)
        self.name = name if name else re.sub('SamplingGroup$', '', cls.__name__)
        self.args, self.kwargs = args, kwargs

    def make_group(self, sites: tp.Iterable[_Site]) -> SamplingGroup:
        return self.cls((site for site in sites if self.match.match(site['name'])),
                        self.name, *self.args, **self.kwargs)

    def __repr__(self):
        return f'<{type(self).__name__}(name={self.name}, match={self.match}, cls={self.cls})>'


class GroupCollection(torch.nn.Module):
    def __init__(self, specs: tp.Iterable[GroupSpec], trace: pyro.poutine.Trace):
        super().__init__()

        sites = [site for name, site in trace.iter_stochastic_nodes()]
        for spec in specs:
            group = spec.make_group(sites)
            for site in group.sites.values():
                sites.remove(site)
            setattr(self, spec.name, group)
        if sites:
            setattr(self, 'Default', GroupSpec().make_group(sites))

    def __iter__(self) -> tp.Iterable[SamplingGroup]:
        return self.children()


class Guide(PyroModule):
    def __init__(self, *specs: GroupSpec, model=None, name=''):
        super().__init__(name)

        self.specs: tp.Iterable[GroupSpec] = specs
        self.model: tp.Callable = model

        self.prototype_trace: tp.Optional[pyro.poutine.Trace] = None
        self.groups: tp.Optional[GroupCollection] = None

        self.frames: tp.MutableMapping[str, CondIndepStackFrame] = {}
        self.plates: tp.MutableMapping[str, pyro.plate] = {}

    def _setup_prototype(self, *args, **kwargs):
        # run the model so we can inspect its structure
        with poutine.block():
            with poutine.trace() as trace:
                with poutine.block(hide_fn=prototype_hide_fn):
                    with init_msgr:
                        self.model(*args, **kwargs)
        self.prototype_trace = trace.trace

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            for frame in site["cond_indep_stack"]:
                if not frame.vectorized:
                    raise NotImplementedError("EasyGuide does not support sequential pyro.plate")
                self.frames[frame.name] = frame

        self.groups = GroupCollection(self.specs, self.prototype_trace)

    def plate(self, name, size=None, subsample_size=None, subsample=None, *args, **kwargs):
        if name not in self.plates:
            self.plates[name] = pyro.plate(name, size, subsample_size, subsample, *args, **kwargs)
        return self.plates[name]

    def guide(self, *args, **kwargs) -> tp.Dict[str, torch.Tensor]:
        # Union of all model samples dicts from self.groups
        return dict(item
                    for group in self.groups
                    for item in group(*args, **kwargs)[1].items())

    def forward(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)
        result = self.guide(*args, **kwargs)
        self.plates.clear()
        return result

    def __getstate__(self):
        # Don't pickle the model
        state = self.__dict__.copy()
        state['model'] = None
        return state


register_globals(**{a: globals()[a] for a in __all__ if a in globals()})
