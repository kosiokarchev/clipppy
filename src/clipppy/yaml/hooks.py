from __future__ import annotations

from ruamel.yaml import MappingNode, Node, ScalarNode, SequenceNode

from .constructor import ClipppyConstructor
from ..stochastic.stochastic import StochasticSpecs
from ..utils import Sentinel


def stochastic_specs_hook(node: Node, constructor: ClipppyConstructor):
    if isinstance(node, MappingNode):
        seq_tag = constructor.resolver.resolve(SequenceNode, [], (True, False))
        return SequenceNode(tag=node.tag, anchor=node.anchor, value=[SequenceNode(
            tag=seq_tag, value=[
                SequenceNode(tag=seq_tag, value=[
                    ScalarNode(tag=constructor._get_py_name(repr(Sentinel.merge)), value='')
                    if constructor.resolver.is_merge(key) else key,
                    val
                ]) for key, val in node.value
            ]
        )])


ClipppyConstructor.add_type_hook(StochasticSpecs, stochastic_specs_hook)
