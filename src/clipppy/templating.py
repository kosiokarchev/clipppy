import re
import string
import typing
from collections import ChainMap
from functools import partial

from frozendict import frozendict

_sentinel_dict = frozendict()

class Template(string.Template):
    def safe_convert(self, mo: re.Match, mapping: typing.Mapping[str, str]):
        named = mo.group('named') or mo.group('braced')
        if named is not None:
            try:
                return str(mapping[named])
            except KeyError:
                return mo.group()
        if mo.group('escaped') is not None:
            return self.delimiter
        if mo.group('invalid') is not None:
            return mo.group()
        raise ValueError('Unrecognized named group in pattern', self.pattern)

    @staticmethod
    def get_mapping(mapping, **kwargs):
        return kwargs if mapping is _sentinel_dict else ChainMap(kwargs, mapping)

    def safe_substitute(self, mapping=_sentinel_dict, **kws):
        # Helper function for .sub()
        return self.pattern.sub(partial(self.safe_convert, mapping=self.get_mapping(mapping, **kws)), self.template)


class TemplateWithDefaults(Template):
    braceidpattern = Template.idpattern + r')(\s*(?:|=\s*(?P<opening_brace>\()?(?P<default>([^\\]|\\(\\|\)))*?)(?(opening_brace)\)|)\s*)\s*'

    def safe_convert(self, mo: re.Match, mapping: typing.Mapping[str, str]):
        if mo.group('default') is not None:
            mapping = ChainMap(mapping, {mo.group('braced'): mo.group('default').replace(r'\)', ')').replace(r'\\', '\\')})
        return super().safe_convert(mo, mapping)
