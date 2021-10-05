from __future__ import annotations

import re
from typing import Iterable, Optional, Type

from . import sampling_group
from .sampling_groups.delta import DeltaSamplingGroup
from ..utils import _allmatch
from ..utils.typing import _AnyRegexable, _Site, AnyRegex


class GroupSpec:
    def __init__(self, cls: Type[sampling_group.SamplingGroup] = DeltaSamplingGroup,
                 match: _AnyRegexable = AnyRegex(_allmatch),  # by default include anything
                 exclude: _AnyRegexable = AnyRegex(),         # by default exclude nothing
                 name='',
                 *args, **kwargs):
        self.cls: Type[sampling_group.SamplingGroup] = cls
        assert issubclass(self.cls, sampling_group.SamplingGroup)

        self.match = AnyRegex.get(match)
        self.exclude = AnyRegex.get(exclude)
        self.name = name if name else re.sub('SamplingGroup$', '', cls.__name__)
        self.args, self.kwargs = args, kwargs

    def make_group(self, sites: Iterable[_Site]) -> Optional[sampling_group.SamplingGroup]:
        matched = [site for site in sites
                   if self.match.match(site['name'])
                   and not self.exclude.match(site['name'])]
        return self.cls(matched, self.name, *self.args, **self.kwargs) if matched else None

    def __repr__(self):
        return f'<{type(self).__name__}(name={self.name}, match={self.match}, cls={self.cls})>'
