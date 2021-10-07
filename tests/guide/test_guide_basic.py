from operator import itemgetter

import pyro
from torch import tensor

from clipppy.guide import GroupSpec, Guide
from clipppy.utils.typing import AnyRegex

from tests.guide._guide_utils import unit_normal, some_uniform


def test_setup():
    def model():
        pyro.sample(name, unit_normal)

    guide = Guide(GroupSpec(name='g'), model=model)

    pyro.clear_param_store()
    name = 'a'
    guide.setup()
    g = guide.g
    assert guide.g.sites.keys() == {'a'}

    pyro.clear_param_store()
    name = 'b'
    assert guide.setup() == {'g': g}
    assert guide.g.sites.keys() == {'b'}


class TestGrouping:
    keys = set('abcdefgh')

    @classmethod
    def model(cls):
        for l in cls.keys:
            pyro.sample(l, unit_normal)

    def _get_matched_sites(self, gs: GroupSpec):
        guide = Guide(gs, model=self.model)
        guide.setup()
        return next(iter(guide.children())).sites.keys()

    def test_grouping(self):
        assert self._get_matched_sites(GroupSpec()) == self.keys
        assert self._get_matched_sites(GroupSpec(
            match='|'.join(m := set('abc'))
        )) == self._get_matched_sites(GroupSpec(
            match=m
        )) == self._get_matched_sites(GroupSpec(
            match=AnyRegex(*m)
        )) == m

        assert self._get_matched_sites(GroupSpec(
            match='[a-e]', exclude=('b', 'c')
        )) == self._get_matched_sites(GroupSpec(
            exclude='[^ade]'
        )) == set('ade')


def test_init():
    def model():
        pyro.sample('a', unit_normal, infer=dict(init=tensor(42.)))
        pyro.sample('b', unit_normal, infer=dict(init=26.))
        pyro.sample('c', unit_normal.expand_by((7,)))
        pyro.sample('d', some_uniform)

    pyro.clear_param_store()
    guide = Guide(GroupSpec(name='g'), model=model)
    guide.setup(init={'c': (c := (42**2 + 26**2)**0.5)})
    assert itemgetter('a', 'b')(res := guide()) == (42, 26)
    assert (res['c'] == c).all()
    assert some_uniform.low <= res['d'] <= some_uniform.high
