from itertools import product, chain
from typing import Iterable, Collection

import numpy as np
import torch
from shapely import geometry as sh, ops as shops
from matplotlib.path import Path
from more_itertools import bucket, only, last
from torch import Tensor

from ...utils import torch_get_default_device


def mplpaths_to_polygons(paths: Iterable[Path]) -> Iterable[sh.Polygon]:
    import networkx as nx

    rings = tuple(sh.LinearRing(p) for path in paths for p in path.cleaned().to_polygons())
    polys = tuple(map(sh.Polygon, rings))

    g = nx.DiGraph()
    g.add_nodes_from(range(len(polys)))
    g.add_edges_from((i1, i2) for (i1, p1), (i2, p2) in product(enumerate(polys), enumerate(polys)) if i1 != i2 and p1.contains(p2))

    finalpolys = []
    while g:
        for subnodes in nx.connected_components(g.to_undirected()):
            sg: nx.DiGraph = g.subgraph(subnodes)
            b = bucket(subnodes, sg.in_degree)
            shell = only(b[0])
            holes = list(b[1])
            finalpolys.append((shell, holes))
            g.remove_nodes_from([shell] + holes)
    return (sh.Polygon(rings[shell], holes=[rings[h] for h in holes])
            for shell, holes in finalpolys)


def triangulate_polygon(poly: sh.Polygon) -> Iterable[sh.Polygon]:
    triangulation = shops.triangulate(poly)
    for ring in (poly.exterior, *poly.interiors):
        triangulation = shops.split(sh.MultiPolygon(triangulation), sh.LineString(ring.coords))
    for tri in triangulation.geoms:
        if poly.contains(tri):
            if len(tri.exterior.coords) != 4:  # closed triangle => 3 + 1
                yield from shops.triangulate(tri)
            else:
                yield tri


def contour_to_simplices(paths: Iterable[Path], device=None, dtype=None) -> tuple[Tensor, Tensor]:
    triangulation = tuple(chain.from_iterable(map(
        triangulate_polygon, mplpaths_to_polygons(paths))))
    return (
        torch.from_numpy(np.stack([np.transpose(t.exterior.xy)[:3] for t in triangulation], 0)).to(
            device=device or torch_get_default_device(), dtype=dtype or torch.get_default_dtype()
        ),
        torch.tensor([t.area for t in triangulation], device=device, dtype=dtype)
    )
