import re
import typing as tp

from . import sampling_group
from .sampling_groups import DeltaSamplingGroup
from .. import guide
from ..utils import _allmatch, _nomatch
from ..utils.typing import _Site


class GroupSpec:
    def __init__(self, cls: tp.Type['sampling_group.SamplingGroup'] = DeltaSamplingGroup,
                 match: tp.Union[str, re.Pattern] = _allmatch,  # by default include anything
                 exclude: tp.Union[str, re.Pattern] = _nomatch,  # by default exclude nothing
                 name='',
                 *args, **kwargs):
        if isinstance(cls, str):
            cls = getattr(guide, cls) or getattr(guide, cls+'SamplingGroup', cls)

        self.cls: tp.Type[sampling_group.SamplingGroup] = cls
        assert issubclass(self.cls, sampling_group.SamplingGroup)

        self.match = re.compile(match)
        self.exclude = re.compile(exclude)
        self.name = name if name else re.sub('SamplingGroup$', '', cls.__name__)
        self.args, self.kwargs = args, kwargs

    def make_group(self, sites: tp.Iterable[_Site]) -> tp.Optional['sampling_group.SamplingGroup']:
        matched = [site for site in sites
                   if self.match.match(site['name']) and not self.exclude.match(site['name'])]
        return self.cls(matched, self.name, *self.args, **self.kwargs) if matched else None

    def __repr__(self):
        return f'<{type(self).__name__}(name={self.name}, match={self.match}, cls={self.cls})>'
