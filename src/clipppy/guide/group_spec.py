import re
import typing as tp

from clipppy.globals import _allmatch, _nomatch, _Site, get_global
from clipppy.guide.sampling_group import SamplingGroup
from clipppy.guide.sampling_groups import DeltaSamplingGroup


class GroupSpec:
    def __init__(self, cls: tp.Type[SamplingGroup] = DeltaSamplingGroup,
                 match: tp.Union[str, re.Pattern] = _allmatch,  # by default include anything
                 exclude: tp.Union[str, re.Pattern] = _nomatch,  # by default exclude nothing
                 name='',
                 *args, **kwargs):
        if isinstance(cls, str):
            cls = get_global(cls, get_global(cls+'SamplingGroup', cls, globals()), globals())

        self.cls: tp.Type[SamplingGroup] = cls
        assert issubclass(self.cls, SamplingGroup)

        self.match = re.compile(match)
        self.exclude = re.compile(exclude)
        self.name = name if name else re.sub('SamplingGroup$', '', cls.__name__)
        self.args, self.kwargs = args, kwargs

    def make_group(self, sites: tp.Iterable[_Site]) -> tp.Optional[SamplingGroup]:
        matched = [site for site in sites
                   if self.match.match(site['name']) and not self.exclude.match(site['name'])]
        return self.cls(matched, self.name, *self.args, **self.kwargs) if matched else None

    def __repr__(self):
        return f'<{type(self).__name__}(name={self.name}, match={self.match}, cls={self.cls})>'
