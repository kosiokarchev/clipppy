from __future__ import annotations

from functools import singledispatch
from itertools import repeat
from math import factorial
from operator import itemgetter
from typing import Iterable

import torch
from torch import Tensor, LongTensor
from torch.nn.functional import normalize


def simplex_volume(s: Tensor) -> Tensor:
    return torch.cat(
        (s, s.new_ones(s.shape[:-1] + (1,))), -1
    ).det().abs() / factorial(s.shape[-1])


@singledispatch
def delaunay_idx(pts):
    from scipy.spatial import Delaunay
    return Delaunay(pts).simplices


@delaunay_idx.register
def _(pts: Tensor) -> LongTensor:
    return torch.from_numpy(delaunay_idx(pts.detach().cpu().numpy())).to(device=pts.device, dtype=int)


def delaunay(pts: Tensor) -> Tensor:
    return pts[delaunay_idx(pts)]


def _get_iterator(iterator=None):
    if iterator is None:
        from tqdm.auto import tqdm
        iterator = tqdm
    return iterator


def cells_to_simplices(cells: Iterable[Tensor], norm=True, iterator=None) -> tuple[Tensor, Tensor, list[int]]:
    iterator = _get_iterator(iterator)
    simplices = []
    sweights = []
    indices = []
    for i, cell in enumerate(iterator(cells)):
        simplices.append(s := delaunay(cell))
        vols = simplex_volume(s)
        sweights.append(normalize(vols, p=1, dim=-1, eps=0) if norm else vols)
        indices.extend(repeat(i, len(s)))
    return torch.concatenate(simplices, -3), torch.concatenate(sweights, -1), indices


def points_to_simplices_voronoi(pts: Tensor, weights: Tensor, iterator=None) -> tuple[Tensor, Tensor]:
    from scipy.spatial import Voronoi

    voronoi = Voronoi(pts.detach().cpu().numpy())
    idx = dict(zip(voronoi.point_region, range(len(pts))))
    regions, ridx = zip(*((region, i) for i, region in enumerate(voronoi.regions) if region and -1 not in region))

    simplices, sweights, i = cells_to_simplices(
        (torch.from_numpy(voronoi.vertices[region]).to(pts) for region in regions),
        iterator=iterator
    )

    return simplices, sweights * weights[itemgetter(*itemgetter(*i)(ridx))(idx),]


def points_to_simplices(pts: Tensor, weights: Tensor, iterator=None) -> tuple[Tensor, Tensor]:
    from phytorchx import broadcast_cat

    idx = delaunay_idx(pts)
    vol = simplex_volume(pts[idx])
    simplices, sweights, i = cells_to_simplices(
        broadcast_cat((pts.unsqueeze(-2), torch.stack((torch.zeros_like(weights), weights), -1).unsqueeze(-1)), -1)
        [idx].flatten(-3, -2),
        norm=False, iterator=iterator
    )

    return simplices[..., :-1], sweights / vol[i]
