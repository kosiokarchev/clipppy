from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack
from functools import cached_property
from typing import Iterable, Mapping, MutableMapping, TypeVar, Union

import pyro
import torch
from pyro import distributions as dist
from pyro.distributions.constraints import Constraint
from pyro.distributions.transforms import Transform
from pyro.nn import pyro_method, PyroModule, PyroParam, PyroSample
from torch import BoolTensor, Size, Tensor
from torch.distributions import biject_to

from ..utils import to_tensor
from ..utils.pyro import AbstractPyroModuleMeta
from ..utils.messengers import no_grad_msgr
from ..utils.typing import _Site


_Tensor_Type = TypeVar('_Tensor_Type', bound=Tensor)


class SamplingGroup(PyroModule, metaclass=AbstractPyroModuleMeta):
    """"""
    def _process_prototype(self):
        pos = 0
        for name, site in self.sites.items():
            self.supports[name] = support = site['infer'].get('support', site['fn'].support)
            self.transforms[name] = transform = biject_to(support)

            try:  # torch >= 1.8?
                if isinstance(self.transforms[name], torch.distributions.IndependentTransform):
                    self.transforms[name] = self.transforms[name].base_transform
            except AttributeError:
                pass

            self.shapes[name] = shape = torch.Size(site['fn'].batch_shape + site['fn'].event_shape)

            self.inits[name] = init = transform.inv(torch.as_tensor(site['value']).expand(shape))

            # mask = site['infer'].get('mask', site.get('mask', getattr(site['fn'], '_mask', None)))
            # if mask is None:
            #     mask = init.new_full([], True, dtype=torch.bool)
            if (mask := site.get('mask', None)) is None:
                mask = init.new_full([], True, dtype=torch.bool)
            self.masks[name] = mask.expand_as(init)

            self.sizes[name] = size = int(self.masks[name].sum())
            self.poss[name] = pos
            pos += size

    def __init__(self, sites: Iterable[_Site], name='', *args, **kwargs):
        super().__init__(name)

        self.sites = OrderedDict((site['name'], site) for site in sites)
        self.shapes: MutableMapping[str, Size] = {}
        self.sizes: MutableMapping[str, int] = {}
        self.poss: MutableMapping[str, int] = {}
        self.supports: MutableMapping[str, Constraint] = {}
        self.transforms: MutableMapping[str, Transform] = {}

        self.masks: MutableMapping[str, BoolTensor] = {}
        self.inits: MutableMapping[str, Tensor] = {}

        self._process_prototype()

        self.active = True

    def _cat_sites(self, vals: Mapping[str, _Tensor_Type]) -> _Tensor_Type:
        return torch.cat(tuple(vals[name].flatten() for name in self.sites.keys() if name in vals.keys()), dim=-1)

    @cached_property
    def init(self) -> Tensor:
        return self._cat_sites(self.inits)

    @cached_property
    def mask(self) -> BoolTensor:
        return self._cat_sites(self.masks)

    @property
    def event_shape(self) -> Size:
        return Size([sum(self.sizes.values())])

    @abstractmethod
    def _sample(self, infer: dict = None) -> Tensor: ...

    # By default, include constraining transformation's Jacobian factor
    include_det_jac = True

    def unpack_site(self, arr: Tensor, name: str):
        return arr[..., self.poss[name]:self.poss[name]+self.sizes[name]]

    def jacobian(self, guide_z: Tensor, sites: Iterable[str] = None) -> Tensor:
        return self._cat_sites({
            name: tr.log_abs_det_jacobian(z, tr(z)).exp()
            for name in (sites or self.sites.keys())
            for z in [self.unpack_site(guide_z, name)]
            for tr in [self.transforms[name]]
        })

    def _sample_site(self, group_z: Tensor, name: str, fn: dist.TorchDistribution = None):
        zs = self.unpack_site(group_z, name)
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

    def unpack(self, group_z: Tensor, sites: Mapping[str, _Site] = None, guiding=True) -> MutableMapping[str, Tensor]:
        return {
            name: self._sample_site(group_z, name, site['fn'] if guiding else None)
            for name, site in (sites or self.sites).items()
        }

    @property
    def grad_context(self):
        if self.training:
            return torch.enable_grad()
        else:
            ret = ExitStack()
            ret.enter_context(no_grad_msgr)
            ret.enter_context(torch.no_grad())
            return ret

    def forward(self, infer: dict = None) -> tuple[Tensor, MutableMapping[str, Tensor]]:
        with self.grad_context:
            with pyro.poutine.infer_config(
                    config_fn=lambda site: dict(**(infer if infer is not None else {}), is_auxiliary=True)):
                group_z = self._sample(infer)
            return group_z, self.unpack(group_z)

    def extra_repr(self) -> str:  # pragma: no cover
        return f'{len(self.sites)} sites, {self.event_shape}'

    @staticmethod
    def _scale_diagonal(scale: Union[Tensor, float], jac: Tensor):
        return to_tensor(scale).to(jac).expand_as(jac) / jac

    @staticmethod
    def _scale_matrix(scale: Union[Tensor, float], jac: Tensor):
        scale = to_tensor(scale).to(jac)
        if not scale.shape[-2:] == 2*(jac.shape[-1],):
            scale = dist.util.eye_like(jac, jac.shape[-1]) * scale.expand_as(jac).unsqueeze(-2)
        return scale / jac.unsqueeze(-1)


class SamplingGroupWithPrior(SamplingGroup, ABC):
    guide_z: Tensor

    @abstractmethod
    def prior(self): ...

    @PyroSample
    def guide_z(self):
        return self.prior()

    @pyro_method
    def _sample(self, infer=None) -> Tensor:
        return self.guide_z


class LocatedSamplingGroup(SamplingGroup, ABC):
    loc: Tensor

    @PyroParam(event_dim=1)
    def loc(self):
        return self.init[self.mask]


class ScaledSamplingGroup(SamplingGroup, ABC):
    def __init__(self, sites, name='', init_scale: Union[Tensor, float] = 1., *args, **kwargs):
        super().__init__(sites, name, *args, **kwargs)
        self.init_scale = init_scale


class LocatedAndScaledSamplingGroupWithPrior(
        SamplingGroupWithPrior, LocatedSamplingGroup, ScaledSamplingGroup, ABC):
    pass
