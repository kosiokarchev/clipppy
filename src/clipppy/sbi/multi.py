from __future__ import annotations

from collections import OrderedDict
from functools import cached_property, reduce
from typing import Mapping, ClassVar

import attr
import torch
from more_itertools import always_iterable
from torch import Size, Tensor
from torch.distributions import Transform, biject_to, Distribution
from torch.distributions.constraints import Constraint, _Dependent

from phytorchx import broadcast_cat

from ._typing import _MultiKT, _KT



def dict_to_vect(d: Mapping[_KT, Tensor], ndims: Mapping[_KT, int]) -> Tensor:
    return broadcast_cat(tuple(
        v.flatten(-ndim) if ndim else v.unsqueeze(-1)
        for k, v in d.items() for ndim in [ndims.get(k, 0)]
    ), -1)


def vect_to_dict(v: Tensor, shapes: Mapping[_KT, Size]) -> Mapping[_KT, Tensor]:
    return OrderedDict(
        (key, val) for i in [0] for key, shape in shapes.items()
        for j in [i+shape.numel()]
        for val in [v[..., i:j].reshape(*v.shape[:-1], *shape)]
        for i in [j]
    )


@attr.s(auto_attribs=True, eq=False)
class ParamPackerMixin:
    param_event_dims: Mapping[_KT, int] = attr.ib(factory=dict, kw_only=True)

    def pack(self, d: Mapping[_KT, Tensor]) -> Tensor:
        return dict_to_vect(d, self.param_event_dims)

    def unpack_like(self, d: Mapping[_KT, Tensor], v: Tensor):
        return vect_to_dict(v, OrderedDict(
            (key, val.shape[val.ndim-self.param_event_dims.get(key, 0):])
            for key, val in d.items()
        ))


@attr.s(auto_attribs=True, eq=False)
class AbstractMultiEvent(ParamPackerMixin):
    keys: _MultiKT
    event_shapes: Mapping[_KT, Size]

    @staticmethod
    def _flatten_cat_shapes(shapes: Mapping[_KT, Size]):
        return Size((sum(map(Size.numel, shapes.values())),))

    @cached_property
    def event_shape(self) -> Size:
        return self._flatten_cat_shapes(self.event_shapes)

    def __attrs_post_init__(self):
        self.event_shapes = OrderedDict((key, self.event_shapes[key]) for key in always_iterable(self.keys))
        self.param_event_dims = {key: len(event_shape) for key, event_shape in self.event_shapes.items()} | self.param_event_dims

    def pack(self, d: Mapping[_KT, Tensor]) -> Tensor:
        return super().pack(OrderedDict((key, d[key]) for key in self.keys))

    def unpack(self, value):
        return vect_to_dict(value, self.event_shapes)


@attr.s(auto_attribs=True, eq=False)
class MultiTransform(AbstractMultiEvent, Transform):
    event_shapes_in: Mapping[_KT, Size] = attr.ib(init=False)
    param_event_dims_in: Mapping[_KT, int] = attr.ib(init=False)

    transforms: Mapping[_KT, Transform]

    _inv = None

    domain = _Dependent(event_dim=1)
    codomain = _Dependent(event_dim=1)

    @property
    def bijective(self):
        return all(self.transforms[key].bijective for key in self.keys)

    def __attrs_post_init__(self):
        self.event_shapes_in = OrderedDict((key, self.transforms[key].inverse_shape(self.event_shapes[key])) for key in self.keys)
        self.param_event_dims_in = {key: len(event_shape) for key, event_shape in self.event_shapes_in.items()}
        super().__attrs_post_init__()

        Transform.__init__(self)

    @cached_property
    def event_shape_in(self) -> Size:
        return self._flatten_cat_shapes(self.event_shapes_in)

    def log_abs_det_jacobian(self, x, y):
        xd = vect_to_dict(x, self.event_shapes_in)
        yd = vect_to_dict(y, self.event_shapes)
        return sum(self.transforms[key].log_abs_det_jacobian(xd[key], yd[key]) for key in self.keys)

    def _call(self, x):
        return dict_to_vect(
            OrderedDict((key, self.transforms[key](val)) for key, val in vect_to_dict(x, self.event_shapes_in).items()),
            self.param_event_dims
        )

    def _inv_call(self, y):
        return dict_to_vect(
            OrderedDict((key, self.transforms[key].inv(val)) for key, val in vect_to_dict(y, self.event_shapes).items()),
            self.param_event_dims_in
        )

    def forward_shape(self, shape):
        assert shape == self.event_shape_in
        return self.event_shape

    def inverse_shape(self, shape):
        assert shape == self.event_shape
        return self.event_shape_in


@attr.s(auto_attribs=True, eq=False)
class MultiConstraint(AbstractMultiEvent, Constraint):
    constraints: Mapping[_KT, Constraint]
    event_dim: ClassVar = 1

    def check(self, value):
        return reduce(torch.logical_and, (self.constraints[key].check(val) for key, val in self.unpack(value).items()))


@biject_to.register(MultiConstraint)
def _biject_to_multi_constraint(mc: MultiConstraint):
    return MultiTransform(
        keys=mc.keys, event_shapes=mc.event_shapes,
        transforms={key: biject_to(c) for key, c in mc.constraints.items()}
    )


@attr.s(auto_attribs=True, eq=False)
class MultiDistribution(AbstractMultiEvent):
    event_shapes: Mapping[_KT, Size] = attr.ib(init=False)
    dists: Mapping[_KT, Distribution]

    def __attrs_post_init__(self):
        self.event_shapes = {key: self.dists[key].event_shape for key in self.keys}
        super().__attrs_post_init__()

        self.support = MultiConstraint(self.keys, self.event_shapes, {key: prior.support for key, prior in self.dists.items()})

    def log_prob(self, x):
        return sum(self.dists[key].log_prob(val) for key, val in self.unpack(x).items())
