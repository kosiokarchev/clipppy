from copy import deepcopy
from functools import partial
from operator import itemgetter

import pyro
import torch
import pyro.distributions as dist
from pytest import fixture, mark
from torch import isclose, Size, tensor

from clipppy.distributions import conundis

torch.set_default_dtype(torch.float64)


def test_concrete_in_all():
    for cls in conundis.ConUnDisMixin._concrete.values():
        assert hasattr(conundis, cls.__name__)


# TODO: test that we are testing all distiributions
#       i.e. that they're all in the entities

unconstrained_entities = (
    dist.Uniform(3., 10.),
    dist.Normal(0., 1.),
    dist.HalfNormal(1.),
    dist.Exponential(3.),
    dist.Cauchy(0., 0.7),
    dist.HalfCauchy(0.7)
    # dist.Gamma(1., 1.)
)
constrained_entities = (
    conundis.Uniform(3., 10.),
    conundis.Normal(0., 1.),
    conundis.HalfNormal(1.),
    conundis.Exponential(3.),
    conundis.Cauchy(0., 0.7),
    conundis.HalfCauchy(0.7),
    # conundis.Gamma(1., 1.)
)


CONSTRAINT = (3., 4.)


def check_constraint(samples, low=None, high=None):
    if low is None:
        low = CONSTRAINT[0]
    if high is None:
        high = CONSTRAINT[1]
    return (low <= samples).all() and (samples < high).all()


# TODO: more conundis tests:
#   - correctness of properties
#   - log_prob
class Test:
    @fixture(params=constrained_entities)
    def d(self, request) -> conundis.conundis_mixin._ConUnDisT:
        return deepcopy(request.param)

    @fixture
    def dc(self, d):
        return d.constrain(*CONSTRAINT)

    def test_icdf(self, dc):
        assert isclose(dc.icdf(tensor(0.)), tensor(dc.constraint_lower))
        assert isclose(dc.icdf(tensor(1.)), tensor(dc.constraint_upper))

    def test_cdf(self, dc):
        assert isclose(dc.cdf(tensor(dc.constraint_lower)), tensor(0.))
        assert isclose(dc.cdf(tensor(dc.constraint_upper)), tensor(1.))

    def test_sample(self, dc):
        s = dc.expand_by(Size((300, 400))).to_event(1).sample((7,))
        assert s.shape == (7, 300, 400)
        assert (dc.constraint_lower <= s).all() and (s < dc.constraint_upper).all()

    def test_plating(self, dc):
        with pyro.poutine.trace() as tracer, pyro.plate('plate', 3):
            pyro.sample('a', dc.expand_by((4,)).to_event())
            pyro.sample('b', dc.expand_by((2, 3, 4)).to_event(1))
        a, b = itemgetter(*'ab')(tracer.trace.nodes)

        assert a['value'].shape == (3, 4)
        assert a['fn'].batch_shape == (3,) and a['fn'].event_shape == (4,)
        assert b['fn'].batch_shape == (2, 3) and b['fn'].event_shape == (4,)


@mark.parametrize('d', unconstrained_entities)
def test_messenger(d):
    with pyro.plate('plate', 3), conundis.ConstrainingMessenger({
        'a': CONSTRAINT, 'b': CONSTRAINT,
        'c': (bc := tensor(((1., 2., 3.), (2., 4., 6.))))
    }), pyro.poutine.trace() as tracer:
        pyro.sample('a', d)
        pyro.sample('b', d.expand_by(Size((2, 3, 4))).to_event(1))
        pyro.sample('c', d.expand_by(Size((3,))))
        pyro.sample('e', dist.Normal(0., 1.))
    a, b, c, e = itemgetter(*'abce')(tracer.trace.nodes)

    for samples in map(itemgetter('value'), (a, b)):
        assert check_constraint(samples)
    assert check_constraint(c['value'], *bc)

    # needs .base-dist here because of plate
    assert all(map(partial(isinstance, a['fn'].base_dist), (conundis.ConUnDisMixin, type(d))))
    assert a['fn'].batch_shape == (3,) and a['value'].shape == (3,)
    assert b['fn'].batch_shape == (2, 3) and b['fn'].event_shape == (4,)
    assert c['fn'].batch_shape == (3,) and c['value'].shape == (3,)
    # Normals expand by themselves...
    assert isinstance(e['fn'], dist.Normal)
