from __future__ import annotations

from abc import abstractmethod
from contextlib import ExitStack
from functools import lru_cache
from operator import itemgetter
from typing import cast, Iterable, Mapping, MutableMapping, Union

import pyro
import torch
from pyro import distributions as dist
from pyro.distributions import constraints, transforms
from pyro.nn import PyroModule, PyroParam, PyroSample
from torch.distributions import biject_to

from ..utils import to_tensor
from ..utils.pyro import AbstractPyroModuleMeta, no_grad_msgr
from ..utils.typing import _Site


class SamplingGroup(PyroModule, metaclass=AbstractPyroModuleMeta):
    """"""
    def _process_prototype(self):
        for name, site in self.sites.items():
            self.supports[name] = support = site['infer'].get('support', site['fn'].support)
            self.transforms[name] = transform = biject_to(support)

            try:  # torch >= 1.8?
                if isinstance(self.transforms[name], torch.distributions.IndependentTransform):
                    self.transforms[name] = self.transforms[name].base_transform
            except AttributeError:
                pass

            self.shapes[name] = shape = site['fn'].batch_shape + site['fn'].event_shape
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
        self.shapes: MutableMapping[str, torch.Size] = {}
        self.sizes: MutableMapping[str, int] = {}
        self.supports: MutableMapping[str, constraints.Constraint] = {}
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
    def _sample(self, infer: dict = None) -> torch.Tensor: ...

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

    def unpack_site(self, group_z, pos: int, name: str, fn: dist.TorchDistribution = None):
        zs = group_z[..., pos:pos+self.sizes[name]]
        z = self.inits[name].expand(zs.shape[:-1] + self.masks[name].shape).clone()
        z[..., self.masks[name]] = zs

        transform = self.transforms[name]
        x = transform(z)

        if fn is not None:
            if self.include_det_jac and transform.bijective:
                log_density = transform.inv.log_abs_det_jacobian(x, z)
                log_density = log_density.sum(list(range(-(log_density.ndim - z.ndim + fn.event_dim), 0)))
            else:
                log_density = 0.

            pyro.sample(name, dist.Delta(x, log_density=log_density, event_dim=fn.event_dim))  # , extra_event_dim=len(fn.batch_shape)))

        return x

    def unpack(self, group_z: torch.Tensor, sites: Mapping[str, _Site] = None, guiding=True) -> MutableMapping[str, torch.Tensor]:
        model_zs = {}
        pos = 0
        for name, fn in map(itemgetter('name', 'fn'), (sites or self.sites).values()):
            model_zs[name] = self.unpack_site(group_z, pos, name, fn if guiding else None)
            pos += self.sizes[name]
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

    def forward(self, infer: dict = None) -> tuple[torch.Tensor, MutableMapping[str, torch.Tensor]]:
        with self.grad_context:
            with pyro.poutine.infer_config(
                    config_fn=lambda site: dict(**(infer if infer is not None else {}), is_auxiliary=True)):
                group_z = self._sample(infer)
            return group_z, self.unpack(group_z)

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
    def _sample(self, infer=None) -> torch.Tensor:
        return self.guide_z
