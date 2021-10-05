import pyro
from pytest import fixture
from torch import Size, tensor
from torch.distributions import biject_to
from torch.distributions.constraints import interval, positive, real

from clipppy.guide import DeltaSamplingGroup, GroupSpec, Guide

from tests.guide._guide_utils import ClearParamStore, some_uniform, unit_normal


class TestSampling(ClearParamStore):
    @staticmethod
    def model_ab():
        pyro.sample('a', unit_normal)
        pyro.sample('b', some_uniform)

    @fixture(scope='function')
    def guide_ab(self):
        guide = Guide(GroupSpec(name='g'), model=self.model_ab)
        guide.setup()
        return guide

    def test_group_active(self):
        guide = Guide(GroupSpec(name='a', match='a'), GroupSpec(name='b', match='b'), model=self.model_ab)
        assert guide().keys() == {'a', 'b'}
        guide.b.active = False
        assert guide().keys() == {'a'}

    def test_add_noise(self, guide_ab):
        assert guide_ab() == guide_ab()

        guide_ab.add_noise = {'a': 0.1}
        ress = guide_ab(), guide_ab()
        assert ress[0]['a'] != ress[1]['a'] and ress[0]['b'] == ress[1]['b']

    def test_support(self, guide_ab):
        def model():
            pyro.sample('a', unit_normal)
            pyro.sample('b', unit_normal, infer={'support': positive, 'init': tensor(42.)})
            pyro.sample('c', some_uniform)

        guide = Guide(GroupSpec(name='g'), model=model)
        guide.setup()
        g: DeltaSamplingGroup = guide.g

        assert g.supports['a'] is real
        assert g.supports['b'] is positive
        # interval does not support proper comparison and support is a dynamic property on Uniform...
        assert (isinstance(g.supports['c'], interval)
                and g.supports['c'].lower_bound == some_uniform._unbroadcasted_low
                and g.supports['c'].upper_bound == some_uniform._unbroadcasted_high)

        for name in g.sites.keys():
            assert g.transforms[name] == biject_to(g.supports[name])

        assert all(g.supports[name].check(val) for name, val in guide().items())

    def test_sizes(self):
        _size = Size((7, 8))

        def model():
            pyro.sample('a', unit_normal)
            pyro.sample('b', unit_normal.expand_by(_size).to_event(1))

        guide = Guide(GroupSpec(name='g'), model=model)
        guide.setup()
        g: DeltaSamplingGroup = guide.g
        assert g.sizes == {'a': 1, 'b': _size.numel()}
        assert g.init.shape == g.event_shape == (_size.numel()+1,)
        assert guide()['a'].shape == () and guide()['b'].shape == _size

    def test_grad_context(self, guide_ab):
        guide_ab.train()
        assert all(t.requires_grad for t in guide_ab().values())

        guide_ab.eval()
        assert all(not t.requires_grad for t in guide_ab().values())
