import re

from clipppy.utils.typing import AnyRegex


def test_init():
    assert AnyRegex('a', re.compile('b')).patterns == (re.compile('a'), re.compile('b'))
    assert AnyRegex.get(a := AnyRegex()) is a
    assert AnyRegex.get('a', re.compile('b')).patterns == (re.compile('a'), re.compile('b'))
    assert AnyRegex.get('a', re.compile('b'), ('c', ('d', 'e'))).patterns == tuple(
        map(re.compile, 'abcde'))


def test_match():
    r = AnyRegex('a', 'b')
    assert r.match('a') and r.match('b') and not r.match('c')
