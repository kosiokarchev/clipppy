from __future__ import annotations

import re
from functools import partial
from typing import Type, Union

from ruamel.yaml import MappingNode, Node, ScalarNode, SequenceNode, VersionedResolver

from .constructor import ClipppyConstructor
from ..clipppy import Clipppy
from ..utils import Sentinel


def _key_is(key: Union[Node, Sentinel], spval: Sentinel, tag: str):
    return key is spval or isinstance(key, Node) and key.tag == tag


class ClipppyResolver(VersionedResolver):
    _pos_tag = 'tag:clipppy:pos'
    _mergepos_tag = 'tag:clipppy:mergepos'
    _merge_tag = 'tag:yaml.org,2002:merge'

    is_pos = partial(_key_is, spval=Sentinel.pos, tag=_pos_tag)
    is_mergepos = partial(_key_is, spval=Sentinel.mergepos, tag=_mergepos_tag)
    is_merge = partial(_key_is, spval=Sentinel.merge, tag=_merge_tag)

    # bugfix
    def resolve(self, kind: Type[Node], value, implicit: tuple[bool, bool]):
        if issubclass(kind, ScalarNode) and implicit[0]:
            if value == "":
                resolvers = self.versioned_resolver.get("", [])
            else:
                resolvers = self.versioned_resolver.get(value[0], [])
            resolvers += self.versioned_resolver.get(None, [])
            for tag, regexp in resolvers:
                if regexp.match(value):
                    return tag
        if self.yaml_path_resolvers and self.resolver_exact_paths:
            exact_paths = self.resolver_exact_paths[-1]
            if kind in exact_paths:
                return exact_paths[kind]
            if None in exact_paths:
                return exact_paths[None]
        if issubclass(kind, ScalarNode):
            return self.DEFAULT_SCALAR_TAG
        elif issubclass(kind, SequenceNode):
            return self.DEFAULT_SEQUENCE_TAG
        elif issubclass(kind, MappingNode):
            return self.DEFAULT_MAPPING_TAG


ClipppyResolver.add_implicit_resolver(ClipppyResolver._pos_tag, re.compile('/'), '/')
ClipppyResolver.add_implicit_resolver(ClipppyResolver._mergepos_tag, re.compile('<'), '<')


class ImplicitClipppyResolver(ClipppyResolver):
    pass


ImplicitClipppyResolver.add_path_resolver(ClipppyConstructor.type_to_tag[Clipppy], [])
