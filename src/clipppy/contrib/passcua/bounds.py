from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from numbers import Real
from typing import Union

import torch
from torch import Tensor, BoolTensor, Size
from torch.linalg import vecdot, vector_norm as norm, inv, cholesky
from typing_extensions import Self


def matvec(M, v):
    return (M @ v.unsqueeze(-1)).squeeze(-1)


def random_direction(shape, ndim, **kwargs):
    return (d := torch.randn(shape+(ndim,), **kwargs)) / norm(d, dim=-1, keepdim=True)


@dataclass
class Bound(ABC):
    """Convex n-dimensional volume."""

    @abstractmethod
    def intersection_dist(self, p: Tensor, d: Tensor) -> Tensor: ...

    def intersect(self, p: Tensor, d: Tensor) -> Tensor:
        return p + self.intersection_dist(p, d).unsqueeze(-1) * d

    def contains(self, p: Tensor) -> Union[BoolTensor, Tensor]:
        raise NotImplementedError

    def get_point(self) -> Tensor:
        raise NotImplementedError

    def uniform(self, shape: Size, **kwargs) -> Tensor:
        raise NotImplementedError

    @classmethod
    def fit(cls, points: Tensor, *args, **kwargs) -> Self:
        raise NotImplementedError


@dataclass
class TranslatedBound(Bound, ABC):
    m: Tensor

    @cached_property
    def ndim(self):
        return self.m.shape[-1]

    @classmethod
    def new_nd(cls, n, dtype=None, device=None, **kwargs):
        return cls(torch.zeros(n, dtype=dtype, device=device), **kwargs)

    def get_point(self) -> Tensor:
        return self.m

    def normalise(self, x: Tensor, shift: bool):
        return x-self.m if shift else x

    def denormalise(self, x: Tensor) -> Tensor:
        return x+self.m


@dataclass
class UnitCube(TranslatedBound):
    """Cube spanning [-1, 1] along every axis."""

    def intersection_dist(self, p: Tensor, d: Tensor) -> Tensor:
        p, d = self.normalise(p, True), self.normalise(d, False)

        return ((p.new_ones(p.shape[-1]) - p * torch.where(d>=0, 1, -1)) / d.abs()).amin(-1)

    def contains(self, p: Tensor) -> Union[BoolTensor, Tensor]:
        return (self.normalise(p, True).abs() < 1).all(dim=-1)

    def uniform(self, shape: Size, **kwargs) -> Tensor:
        return self.denormalise(2*torch.rand(shape+(self.ndim,), **kwargs) - 1)


@dataclass
class Cuboid(UnitCube):
    A: Tensor = None
    inv_A: Tensor = None

    def __post_init__(self):
        if self.inv_A is None:
            self.inv_A = inv(self.A)

        if self.A is None:
            self.A = inv(self.inv_A)

    def normalise(self, x: Tensor, shift: bool):
        return matvec(self.inv_A, super().normalise(x, shift))

    def denormalise(self, x: Tensor) -> Tensor:
        return super().denormalise(matvec(self.A, x))

    @classmethod
    def from_extents(cls, low: Tensor, high: Tensor) -> Self:
        return cls((high+low) / 2, ((high-low) / 2).diag_embed())


@dataclass
class UnitSphere(TranslatedBound):
    def intersection_dist(self, p: Tensor, d: Tensor) -> Tensor:
        p, d = self.normalise(p, True), self.normalise(d, False)

        p2 = vecdot(p, p, dim=-1)
        d2 = vecdot(d, d, dim=-1)
        pd = vecdot(p, d, dim=-1)

        return (-pd + (pd*pd - d2*(p2-1))**0.5) / d2

    def contains(self, p: Tensor) -> Union[BoolTensor, Tensor]:
        return norm(self.normalise(p, True), dim=-1) < 1

    def uniform(self, shape: Size, **kwargs) -> Tensor:
        return self.denormalise(
            torch.rand(shape, **kwargs).pow_(1/self.ndim) *
            random_direction(shape, self.ndim, **kwargs)
        )


@dataclass
class Sphere(UnitSphere):
    r: Union[Tensor, Real] = 1.

    def normalise(self, x: Tensor, shift: bool):
        return super().normalise(x, shift) / self.r

    def denormalise(self, x: Tensor) -> Tensor:
        return super().denormalise(self.r * x)


@dataclass
class Ellipsoid(UnitSphere):
    R2: Tensor = None
    inv_R: Tensor = None

    def __post_init__(self):
        if self.inv_R is None:
            self.inv_R = inv(cholesky(self.R2))

        self.R = inv(self.inv_R)

        if self.R2 is None:
            self.R2 = inv(self.inv_R.T @ self.inv_R)

    def normalise(self, x: Tensor, shift: bool):
        return matvec(self.inv_R, super().normalise(x, shift))

    def denormalise(self, x: Tensor) -> Tensor:
        return super().denormalise(matvec(self.R, x))

    @classmethod
    def fit(cls, points: Tensor, tol=1e-3, pad=0, validate=True, **kwargs) -> Self:
        # Modified from https://gist.github.com/jasnyder/ccdf5ca4e76e81a2047c78887a95e0a2
        # See also https://link.springer.com/article/10.1007/s12532-023-00242-8

        N, d = points.shape

        Q = torch.cat((points, points.new_ones(N, 1)), -1).T
        u = points.new_full((N,), 1/N)

        err = tol+1.0
        while err > tol:
            Mmax, idx = (Q.T @ inv(Q*u @ Q.T) * Q.T).sum(-1).max(dim=-1)

            step_size = (Mmax-1-d) / ((d+1) * (Mmax-1))
            new_u = (1-step_size) * u
            new_u[idx] += step_size
            err = norm(new_u-u)
            u = new_u

        ret = cls(
            c := u @ points,
            (1+pad)**2 * d * ((points.T * u) @ points - torch.outer(c, c))
        )

        if validate:
            maxndist = norm(ret.normalise(points, True), dim=-1).max()
            if not maxndist <= 1:
                ret = cls(ret.m, ((1+pad)*maxndist)**2 * ret.R2)

        return ret
