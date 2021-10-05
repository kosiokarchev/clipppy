import pyro
from pytest import mark
from torch import tensor
from torch.distributions.constraints import positive

from clipppy.guide import (DeltaSamplingGroup, DiagonalNormalSamplingGroup,
                           GroupSpec, Guide, HierarchicPartialMultivariateNormalSamplingGroup,
                           MultivariateNormalSamplingGroup,
                           PartialMultivariateNormalSamplingGroup)

from tests.guide._guide_utils import ClearParamStore, some_uniform, unit_normal


class TestSamplingGroups(ClearParamStore):
    @mark.parametrize('cls, kwargs', (
        (DeltaSamplingGroup, {}),
        (DiagonalNormalSamplingGroup, {}),
        (MultivariateNormalSamplingGroup, {}),
        (PartialMultivariateNormalSamplingGroup,
         dict(diag=('c', 'd'))),
        (HierarchicPartialMultivariateNormalSamplingGroup,
         dict(diag=('c', 'd'), hdims={'c': 1, 'd': 1}))
    ))
    def test_sampling_groups_basic(self, cls, kwargs):
        shapes = {'a': (), 'b': (3,), 'c': (3, 4, 5), 'd': (3, 4, 6)}

        def model():
            pyro.sample('a', unit_normal, infer={'support': positive, 'init': tensor(42.)})
            pyro.sample('b', unit_normal.expand_by(shapes['b']).to_event(None))
            pyro.sample('c', some_uniform.expand_by(shapes['c']).to_event(None))
            pyro.sample('d', unit_normal.expand_by(shapes['d']).to_event(None))

        guide = Guide(GroupSpec(cls, **kwargs, name='g'), model=model)

        assert all(val.shape == shapes[name] for name, val in guide().items())

        with pyro.plate('plate', 26):
            res = guide()
        assert all(val.shape == (26,) + shapes[name] for name, val in res.items())
        assert all(guide.g.supports[name].check(val).all() for name, val in res.items())
