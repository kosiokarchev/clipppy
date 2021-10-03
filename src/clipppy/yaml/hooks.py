from __future__ import annotations

from ruamel.yaml import MappingNode, Node, SequenceNode

from .constructor import ClipppyConstructor
from ..stochastic.stochastic import StochasticSpecs
from ..utils import Sentinel


def stochastic_specs_hook(node: Node, constructor: ClipppyConstructor):
    if isinstance(node, MappingNode):
        for i, (key, val) in enumerate(node.value):
            key: Node
            if constructor.resolver.is_merge(key):
                node.value[i] = (Sentinel.merge, val)
            elif isinstance(key, SequenceNode) and constructor.is_default_tagged(key):
                node.value[i] = (constructor.construct_object(key), val)
        return SequenceNode(tag=node.tag, anchor=node.anchor, value=[MappingNode(
            tag=constructor.resolver.resolve(type(node), node.value, (True, False)),
            value=node.value
        )])


ClipppyConstructor.add_type_hook(StochasticSpecs, stochastic_specs_hook)
