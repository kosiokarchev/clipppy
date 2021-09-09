from __future__ import annotations

from abc import ABCMeta, abstractmethod
from contextlib import ExitStack
from functools import lru_cache
from operator import itemgetter
from typing import cast, Iterable, Mapping, MutableMapping, Union

import pyro
import torch
from pyro import distributions as dist
from pyro.distributions import transforms
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.poutine import runtime
from torch.distributions import biject_to

from . import guide as _guide
from ..utils import enumlstrip, to_tensor
from ..utils.pyro import no_grad_msgr
from ..utils.typing import _Site


class _AbstractPyroModuleMeta(type(PyroModule), ABCMeta):
    """"""
    def __getattr__(self, item):
        if item.startswith('_pyro_prior_'):
            return getattr(self, item.lstrip('_pyro_prior_')).prior
        return super().__getattr__(item)


class SamplingGroup(PyroModule, metaclass=_AbstractPyroModuleMeta):
    """"""
    def _process_prototype(self):
        common_dims = set(f.dim
                          for site in self.sites.values()
                          for f in site['cond_indep_stack']
                          if f.vectorized)
        rightmost_common_dim = max(common_dims, default=-float('inf'))

        for name, site in self.sites.items():
            transform = biject_to(site['infer'].get('support', site['fn'].support))
            self.transforms[name] = transform

            try:  # torch >= 1.8?
                if isinstance(self.transforms[name], torch.distributions.IndependentTransform):
                    self.transforms[name] = self.transforms[name].base_transform
            except AttributeError:
                pass

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

            # mask = site['infer'].get('mask', site.get('mask', getattr(site['fn'], '_mask', None)))
            # if mask is None:
            #     mask = init.new_full([], True, dtype=torch.bool)
            if (mask := site.get('mask', None)) is None:
                mask = init.new_full([], True, dtype=torch.bool)
            self.masks[name] = mask.expand_as(init)

            self.sizes[name] = int(self.masks[name].sum())

    def __init__(self, sites: Iterable[_Site], name='', *args, **kwargs):
        super().__init__(name)

        self.sites: Mapping[str, _Site] = {site['name']: site for site in sites}
        self.event_shapes: MutableMapping[str, torch.Size] = {}
        self.batch_shapes: MutableMapping[str, torch.Size] = {}
        self.sizes: MutableMapping[str, int] = {}
        self.transforms: MutableMapping[str, transforms.Transform] = {}

        self.masks: MutableMapping[str, torch.Tensor] = {}
        self.inits: MutableMapping[str, torch.Tensor] = {}

        self._process_prototype()

        self.active = True

    @property
    @lru_cache()
    def init(self) -> torch.Tensor:
        return torch.cat([t.flatten() for t in self.inits.values()])

    @property
    @lru_cache()
    def mask(self) -> torch.BoolTensor:
        return cast(torch.BoolTensor,
                       torch.cat([m.flatten() for m in self.masks.values()]))

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size([sum(self.sizes.values())])

    @abstractmethod
    def _sample(self, infer) -> torch.Tensor: ...

    # By default, include constraining transformation's Jacobian factor
    include_det_jac = True

    def unpacker(self, arr: torch.Tensor) -> Iterable[torch.Tensor]:
        for pos, size in zip((s for s in [0] for x in self.sizes.values() for s in [x+s]), self.sizes.values()):
            yield arr[..., pos-size:pos]

    def jacobian(self, guide_z: torch.Tensor) -> torch.Tensor:
        return torch.cat(tuple(
            tr.log_abs_det_jacobian(z, tr(z)).exp()
            for z, tr in zip(self.unpacker(guide_z), self.transforms.values())
        ), dim=-1)

    def unpack(self, group_z: torch.Tensor, guide: _guide.Guide = None) -> MutableMapping[str, torch.Tensor]:
        model_zs = {}
        for pos, (name, fn, frames), transform in zip(
            # lazy cumsum!! python ftw!
            (s for s in [0] for x in self.sizes.values() for s in [x+s]),
            map(itemgetter('name', 'fn', 'cond_indep_stack'), self.sites.values()),
            self.transforms.values()
        ):
            fn: dist.TorchDistribution
            zs = group_z[..., pos-self.sizes[name]:pos]
            z = self.inits[name].expand(zs.shape[:-1] + self.masks[name].shape).clone()
            z[..., self.masks[name]] = zs

            x = transform(z)

            if self.include_det_jac and transform.bijective:
                log_density = transform.inv.log_abs_det_jacobian(x, z)
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

    def forward(self, guide: _guide.Guide = None, infer: dict = None) -> tuple[torch.Tensor, MutableMapping[str, torch.Tensor]]:
        with pyro.poutine.infer_config(
                config_fn=lambda site: dict(**(infer if infer is not None else {}), is_auxiliary=True)):
            with self.grad_context:
                group_z = self._sample(infer)
                return group_z, self.unpack(group_z, guide)

    def extra_repr(self) -> str:
        return f'{len(self.sites)} sites, {self.event_shape}'


class LocatedSamplingGroupWithPrior(SamplingGroup):
    def __init__(self, sites, name='', *args, **kwargs):
        super().__init__(sites, name, *args, **kwargs)
        self.loc = PyroParam(self.init[self.mask], event_dim=1)

    @staticmethod
    def _scale_diagonal(scale: Union[torch.Tensor, float], jac: torch.Tensor):
        return to_tensor(scale).to(jac).expand_as(jac) / jac

    @staticmethod
    def _scale_matrix(scale: Union[torch.Tensor, float], jac: torch.Tensor):
        scale = to_tensor(scale).to(jac)
        if not scale.shape[-2:] == 2*(jac.shape[-1],):
            scale = dist.util.eye_like(jac, jac.shape[-1]) * scale.expand_as(jac).unsqueeze(-2)
        return scale / jac.unsqueeze(-1)

    @abstractmethod
    def prior(self): ...

    @PyroSample
    def guide_z(self):
        return self.prior()

    @pyro.nn.pyro_method
    def _sample(self, infer) -> torch.Tensor:
        return self.guide_z
