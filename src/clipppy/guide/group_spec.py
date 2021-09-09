from __future__ import annotations

import re
from typing import Iterable, Optional, Type, Union

from more_itertools import always_iterable

from . import sampling_group
from .sampling_groups import DeltaSamplingGroup
from ..utils import _allmatch, _nomatch
from ..utils.typing import _Regex, _Site


class GroupSpec:
    def __init__(self, cls: Type[sampling_group.SamplingGroup] = DeltaSamplingGroup,
                 match: Union[_Regex, Iterable[_Regex]] = _allmatch,  # by default include anything
                 exclude: Union[_Regex, Iterable[_Regex]] = _nomatch,  # by default exclude nothing
                 name='',
                 *args, **kwargs):
        self.cls: Type[sampling_group.SamplingGroup] = cls
        assert issubclass(self.cls, sampling_group.SamplingGroup)

        self.match = list(map(re.compile, always_iterable(match)))
        self.exclude = list(map(re.compile, always_iterable(exclude)))
        self.name = name if name else re.sub('SamplingGroup$', '', cls.__name__)
        self.args, self.kwargs = args, kwargs

    def make_group(self, sites: Iterable[_Site]) -> Optional[sampling_group.SamplingGroup]:
        matched = [site for site in sites
                   if any(m.match(site['name']) for m in self.match)
                   and not any(e.match(site['name']) for e in self.exclude)]
        return self.cls(matched, self.name, *self.args, **self.kwargs) if matched else None

    def __repr__(self):
        return f'<{type(self).__name__}(name={self.name}, match={self.match}, cls={self.cls})>'
