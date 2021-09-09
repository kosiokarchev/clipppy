from __future__ import annotations

from ruamel.yaml import Node, SequenceNode

from .constructor import ClipppyConstructor
from ..stochastic.stochastic import StochasticSpecs
from ..utils import PseudoString, Sentinel


def stochastic_specs_hook(node: Node, constructor: ClipppyConstructor):
    for i, (key, val) in enumerate(node.value):
        key: Node
        if constructor.resolver.is_merge(key):
            node.value[i] = (PseudoString(Sentinel.merge), val)
        elif isinstance(key, SequenceNode) and constructor.is_default_tagged(key):
            node.value[i] = (PseudoString(constructor.construct_object(key)), val)


ClipppyConstructor.add_type_hook(StochasticSpecs, stochastic_specs_hook)

_ = None  # from .hooks import _  : * does not work in local scopes
