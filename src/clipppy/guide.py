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
from torch.distributions import biject_to

from .globals import get_global, _Site, register_globals, enumlstrip, init_msgr, no_grad_msgr

__all__ = ('DeltaSamplingGroup', 'DiagonalNormalSamplingGroup', 'MultivariateNormalSamplingGroup', 'GroupSpec', 'Guide')


class _AbstractPyroModuleMeta(type(PyroModule), ABCMeta):
    def __getattr__(self, item):
        if item.startswith('_pyro_prior_'):
            return getattr(self, item.lstrip('_pyro_prior_')).prior
        return super().__getattr__(item)


class SamplingGroup(PyroModule, metaclass=_AbstractPyroModuleMeta):
    def _process_prototype(self):
        common_dims = set(f.dim
                          for site in self.sites.values()
                          for f in site['cond_indep_stack']
                          if f.vectorized)
        rightmost_common_dim = max(common_dims, default=-float('inf'))

        for name, site in self.sites.items():
            transform = biject_to(site['infer'].get('support', site['fn'].support))
            self.transforms[name] = transform

            self.event_shapes[name] = site['fn'].event_shape
            batch_shape = site['fn'].batch_shape

            self.batch_shapes[name] = torch.Size(
                enumlstrip(batch_shape, lambda i, x: x == 1 or i-len(batch_shape) in common_dims))
            if len(self.batch_shapes[name]) > -rightmost_common_dim:
                raise ValueError(
                    'grouping expects all per-site plates to be right of all common plates, '
                    f'but found a per-site plate {-len(self.batch_shapes[name])} '
                    f'on left at site {site["name"]!r}')

            shape = self.batch_shapes[name] + self.event_shapes[name]

            init = transform.inv(site['value'].__getitem__((0,)*(site['value'].ndim - len(shape))).expand(shape))

            self.inits[name] = init

            # TODO: Wait for better days to come...:
            # _ if (_:=site.get('mask', None)) is not None....
            mask = site['infer'].get('mask', site.get('mask', getattr(site['fn'], '_mask', None)))
            if mask is None:
                mask = init.new_full([], True, dtype=torch.bool)
            self.masks[name] = mask.expand_as(init)

            self.sizes[name] = int(self.masks[name].sum())

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
    @lru_cache()
    def init(self) -> torch.Tensor:
        return torch.cat([t.flatten() for t in self.inits.values()])

    @property
    @lru_cache()
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
            zs = group_z[..., pos-self.sizes[name]:pos]
            z = self.inits[name].expand(zs.shape[:-1] + self.masks[name].shape).clone()
            z[..., self.masks[name]] = zs

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
                model_zs[name] = pyro.sample(name, delta)

        return model_zs

    @property
    def grad_context(self):
        if self.training:
            return torch.enable_grad()
        else:
            ret = ExitStack()
            ret.enter_context(no_grad_msgr)
            ret.enter_context(torch.no_grad())
            return ret

    def forward(self, guide: 'Guide' = None, infer: dict = None) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor]]:
        with pyro.poutine.infer_config(
                config_fn=lambda site: dict(**(infer if infer is not None else {}), is_auxiliary=True)):
            with self.grad_context:
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

        self.loc = PyroParam(self.init[self.mask], event_dim=1)

    # Emulate EasyGuide's map_estimate implementation
    include_det_jac = False

    def _sample(self, infer) -> torch.Tensor:
        return tp.cast(torch.Tensor, self.loc)


class DiagonalNormalSamplingGroup(SamplingGroup):
    def __init__(self, sites, name='', init_scale=1., *args, **kwargs):
        super().__init__(sites, name, *args, **kwargs)

        self.loc = PyroParam(self.init[self.mask], event_dim=1)
        self.scale = PyroParam(torch.full_like(self.loc, init_scale), event_dim=1,
                               constraint=dist.constraints.positive)

    def prior(self):
        return dist.Normal(self.loc, self.scale).to_event(1)

    @PyroSample
    def guide_z(self):
        return self.prior()

    @pyro.nn.pyro_method
    def _sample(self, infer) -> torch.Tensor:
        return self.guide_z


class MultivariateNormalSamplingGroup(SamplingGroup):
    def __init__(self, sites, name='', init_scale=1., *args, **kwargs):
        super().__init__(sites, name, *args, **kwargs)

        self.loc = PyroParam(self.init[self.mask], event_dim=1)
        self.scale_tril = PyroParam(dist.util.eye_like(self.loc, self.loc.shape[-1]) * init_scale,
                                    event_dim=2, constraint=dist.constraints.lower_cholesky)

    def prior(self):
        return dist.MultivariateNormal(self.loc, scale_tril=self.scale_tril)

    @PyroSample
    def guide_z(self):
        return self.prior()

    @pyro.nn.pyro_method
    def _sample(self, infer) -> torch.Tensor:
        return self.guide_z


class GroupSpec:
    def __init__(self, cls: tp.Type[SamplingGroup] = DeltaSamplingGroup,
                 match: tp.Union[str, re.Pattern] = re.compile('.*'),    # by default include anything
                 exclude: tp.Union[str, re.Pattern] = re.compile('.^'),  # by default exclude nothing
                 name='',
                 *args, **kwargs):
        if isinstance(cls, str):
            cls = get_global(cls, get_global(cls+'SamplingGroup', cls, globals()), globals())

        self.cls: tp.Type[SamplingGroup] = cls
        assert issubclass(self.cls, SamplingGroup)

        self.match = re.compile(match)
        self.exclude = re.compile(exclude)
        self.name = name if name else re.sub('SamplingGroup$', '', cls.__name__)
        self.args, self.kwargs = args, kwargs

    def make_group(self, sites: tp.Iterable[_Site]) -> tp.Optional[SamplingGroup]:
        matched = [site for site in sites
                   if self.match.match(site['name']) and not self.exclude.match(site['name'])]
        return self.cls(matched, self.name, *self.args, **self.kwargs) if matched else None

    def __repr__(self):
        return f'<{type(self).__name__}(name={self.name}, match={self.match}, cls={self.cls})>'


class BaseGuide(PyroModule):
    def __init__(self, model=None, name=''):
        super().__init__(name)

        self.model: tp.Callable = model

        self.prototype_trace: tp.Optional[pyro.poutine.Trace] = None

        self.frames: tp.MutableMapping[str, CondIndepStackFrame] = {}
        self.plates: tp.MutableMapping[str, pyro.plate] = {}

        self.is_setup = False

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

    def setup(self, *args, **kwargs) -> tp.Dict[str, tp.Any]:
        old_children = dict(self.named_children())
        for child in old_children:
            delattr(self, child)

        self.is_setup = True
        self._setup_prototype(*args, **kwargs)
        return old_children

    def plate(self, name, size=None, subsample_size=None, subsample=None, *args, **kwargs):
        if name not in self.plates:
            self.plates[name] = pyro.plate(name, size, subsample_size, subsample, *args, **kwargs)
        return self.plates[name]

    def guide(self, *args, **kwargs) -> tp.Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        if not self.is_setup:
            self.setup(*args, **kwargs)
        result = self.guide(*args, **kwargs)
        self.plates.clear()
        return result

    def __getstate__(self):
        # Don't pickle the model
        state = self.__dict__.copy()
        state['model'] = None
        return state


class Guide(BaseGuide):
    child_spec = SamplingGroup
    children: tp.Callable[[], tp.Iterable[child_spec]]

    def __init__(self, *specs: GroupSpec, model=None, name=''):
        super().__init__(model=model, name=name)

        self.specs: tp.Iterable[GroupSpec] = specs

    def setup(self, *args, **kwargs) -> tp.Dict[str, child_spec]:
        old_children = super().setup(*args, **kwargs)

        sites = [site for name, site in self.prototype_trace.iter_stochastic_nodes()]
        for spec in self.specs:
            group = spec.make_group(sites)
            if group:
                for site in group.sites.values():
                    sites.remove(site)
                setattr(self, spec.name, group)
        # if sites:
        #     setattr(self, 'Default', GroupSpec().make_group(sites))

        return old_children

    def guide(self, *args, **kwargs) -> tp.Dict[str, torch.Tensor]:
        # Union of all model samples dicts from self.children
        return dict(item
                    for group in self.children()
                    for item in group(*args, **kwargs)[1].items())


register_globals(**{a: globals()[a] for a in __all__ if a in globals()})
