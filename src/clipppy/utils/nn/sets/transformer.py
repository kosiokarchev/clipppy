"""
The various components of a Set Transformer, as described in [Lee2019]_.

References
----------
.. [Lee2019] Lee, Juho, et al. "Set transformer: A framework for
             attention-based permutation-invariant neural networks."
             *International conference on machine learning*. PMLR, 2019.
"""

from __future__ import annotations

from typing import Union, Callable, TYPE_CHECKING

import attr
import torch
from torch import Tensor
from torch.nn import Module, Parameter, init, LayerNorm

from ..attrs import AttrsModule, ParametrizedAttrsModel
from ..batched import BatchedMultiheadAttention
from ..empty import _empty_module


@attr.s(auto_attribs=True, eq=False)
class MAB(AttrsModule):
    embed_dim: int
    """Input dimension (i.e. of the space that the set members belong to).
    
    In [Lee2019]_, this is :math:`d`."""

    num_heads: int
    """Number of heads for the ~`torch.nn.MultiheadAttention`. Must divide into `embed_dims`."""

    rFF: Union[Module, Callable[[Tensor], Tensor]] = _empty_module
    """"Row-wise" transform (rFF in eq. (6) of [Lee2019]_):
    ``(batch..., embed_dim) -> (batch..., embed_dim)``."""

    use_layer_norm: bool = True
    """Whether to include layer normalisations."""

    def __attrs_post_init__(self):
        self.mha = BatchedMultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        self.ln_1, self.ln_2 = (
            LayerNorm(self.embed_dim) if self.use_layer_norm else _empty_module
            for _ in range(2))

    def forward(self, x, y) -> Tensor:
        """Multihead Attention Block, eqs. (6, 7) in [Lee2019]_.

        .. note:: Currently, without the ~`torch.nn.LayerNorm`.

        Parameters
        ----------
        x: `~torch.Tensor` ``(batch..., N, embed_dim)``
        y: `~torch.Tensor` ``(batch..., N, embed_dim)``

        Returns
        -------
        `~torch.Tensor` ``(batch..., N, embed_dim)``
        """
        return self.ln_2(
            (h := self.ln_1(
                x + self.mha(x, y, y, need_weights=False)[0]))
            + self.rFF(h))

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class SAB(AttrsModule):
    mab: MAB

    def forward(self, x: Tensor) -> Tensor:
        """Set Attention Block, eq. (8) in [Lee2019]_.

        Parameters
        ----------
        x: `~torch.Tensor` ``(batch..., N, embed_dim)``
           Input `Tensor`. Last dimension must match the `~MAB.embed_dim` of
           the `mab`.

        Returns
        -------
        `~torch.Tensor` ``(batch..., N, embed_dim)``
           Same size as the input.
        """
        return self.mab(x, x)

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class ISAB(ParametrizedAttrsModel):
    m: int
    """Number of inducing points."""

    mab_1: MAB
    r"""First `MAB`: eq. (10) in [Lee2019]_,
    called as :math:`\mathrm{MAB_1}(I, X)`.
    
    .. note:: `embed_dim` must match that of `mab_2`.
    """

    mab_2: MAB
    r"""Second `MAB`: eq. (9) in [Lee2019]_,
    called as :math:`\mathrm{MAB_2}(X, \mathrm{MAB}_1(I, X))`.
    
    .. note:: `embed_dim` must match that of `mab_1`.
    """

    I: Parameter = attr.field(init=False, repr=False)
    """Inducing points with shape ``(m, embed_dim)``."""

    def __attrs_post_init__(self):
        assert self.mab_1.embed_dim == self.mab_2.embed_dim
        self.I = Parameter(torch.empty((self.m, self.mab_1.embed_dim), **self.factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.I)

    def forward(self, x: Tensor) -> Tensor:
        """Induced Set Attention Block, eqs. (9, 10) in [Lee2019]_.

        Parameters
        ----------
        x: `~torch.Tensor` ``(batch..., N, embed_dim)``
           Input `Tensor`. Last dimension must match the `~MAB.embed_dim` of
           the `mab_1` and `mab_2`.

        Returns
        -------
        `~torch.Tensor` ``(batch..., N, embed_dim)``
           Same size as the input.
        """
        return self.mab_2(x, self.mab_1(self.I, x))

    if TYPE_CHECKING:
        __call__ = forward


@attr.s(auto_attribs=True, eq=False)
class PMA(ParametrizedAttrsModel):
    mab: MAB

    k: int = 1
    """Number of seed vectors."""

    rFF: Union[Module, Callable[[Tensor], Tensor]] = _empty_module
    """"Row-wise" transform (rFF in eq. (11) of [Lee2019]_):
        ``(batch..., embed_dim) -> (batch..., embed_dim)``."""

    S: Parameter = attr.field(init=False, repr=False)
    """Seed vectors with shape ``(k, embed_dim)``."""

    def __attrs_post_init__(self):
        self.S = Parameter(torch.empty((self.k, self.mab.embed_dim), **self.factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.S)

    def forward(self, z):
        """Pooling by Multihead Attention, eq. (11) in [Lee2019]_.

        Parameters
        ----------
        z: `~torch.Tensor` ``(batch..., N, embed_dim)``

        Returns
        -------
        `~torch.Tensor` ``(batch..., k, embed_dim)``
            The input pooled onto the `k` seed vectors in `S`.
        """
        return self.mab(self.S, self.rFF(z))

    if TYPE_CHECKING:
        __call__ = forward
