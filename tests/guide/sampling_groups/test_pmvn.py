import pyro
from pytest import fixture
from torch import Size

from clipppy.guide import GroupSpec, Guide, PMVN
from tests.guide._guide_utils import ClearParamStore, unit_normal


class TestPMVN(ClearParamStore):
    shapes = {'a': Size((7,)), 'b': Size(), 'c': Size(),
              'd': Size((26,)), 'e': Size((42,))}

    @classmethod
    def model(cls):
        for name in cls.shapes.keys():
            pyro.sample(name, unit_normal.expand(cls.shapes[name]).to_event(None))

    keys_diag = {'d', 'e'}
    keys_full = set(shapes.keys()) - keys_diag

    @fixture(scope='function')
    def guide(self):
        guide = Guide(GroupSpec(PMVN, diag=self.keys_diag, name='g'), model=self.model)
        guide.setup()
        return guide

    @fixture(scope='function')
    def g(self, guide) -> PMVN:
        return guide.g

    def test_grouping(self, g):
        assert g.sites_full.keys() == self.keys_full
        assert g.sites_diag.keys() == self.keys_diag

    def test_sizes(self, g):
        assert g.size_full == sum(self.shapes[key].numel() for key in self.keys_full)
        assert g.size_diag == sum(self.shapes[key].numel() for key in self.keys_diag)

    def test_sample_full(self, g):
        with pyro.plate('plate', 10):
            res = g.sample_full()

        assert res.keys() == g.sites_full.keys() == self.keys_full
        assert {name: t.shape for name, t in res.items()} == {
            name: (10,) + self.shapes[name] for name in res.keys()
        }
