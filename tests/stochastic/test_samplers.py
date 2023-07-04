import pyro
import torch
from pytest import raises
from pyro.distributions import Normal
from torch.distributions import biject_to
from torch.distributions.constraints import positive

from clipppy.stochastic.sampler import (
    Param, Sampler, Deterministic, Factor,
    PseudoSampler, Context, UnbindEffect, UnsqueezeEffect, MovedimEffect
)
from clipppy.utils import Sentinel


def test_pseudo():
    o = object()
    fo = lambda: o

    assert PseudoSampler(fo)() is o
    assert PseudoSampler(fo, call=True)() is o
    assert PseudoSampler(fo, call=Sentinel.call)() is o
    assert PseudoSampler(o)() is o
    assert PseudoSampler(fo, call=False)() is fo
    assert PseudoSampler(fo, call=Sentinel.no_call)() is fo


def test_context():
    node = pyro.poutine.trace(Context(
        pyro.plate('plate', 100),
        Sampler(Normal(0, 1), name='a')
    )).get_trace().nodes['a']
    assert node['value'].shape == (100,)
    assert node['cond_indep_stack'][0].name == 'plate'


class TestEffects:
    def test_unbind_effect(self):
        assert all(a.shape == (5,) for a in UnbindEffect(torch.rand(5, 6))())
        assert all(a.shape == (6,) for a in UnbindEffect(torch.rand(5, 6), dim=0)())

    def test_unsqueeze_effect(self):
        assert UnsqueezeEffect(torch.rand(5, 6))().shape == (5, 6, 1)
        assert UnsqueezeEffect(torch.rand(5, 6), dim=1)().shape == (5, 1, 6)

    def test_movedim_effect(self):
        assert MovedimEffect(torch.rand(5, 3, 6))().shape == (6, 5, 3)
        assert MovedimEffect(torch.rand(5, 3, 6), source=1, destination=2)().shape == (5, 6, 3)


def test_param():
    pyro.clear_param_store()
    p = Param('a', init=torch.rand(10), support=positive, event_dim=1)
    assert ((v := p()) == pyro.param('a')).all()
    assert (biject_to(positive)(v.unconstrained()) == v).all()


class TestSampler:
    def test_name(self):
        assert Sampler(d=None).name is None
        assert Sampler(d=None, name='a').name == 'a'
        assert Sampler(d=None).set_name('a').name == 'a'
        assert Sampler(d=None, name='a').set_name('b').name == 'b'
        assert Sampler(d=None, name='a').set_name(None).name == 'a'

    def test_infer_dict(self):
        assert Sampler(d=None).infer_dict == {}
        assert (s := Sampler(
            d=None,
            init=torch.tensor(42), mask=torch.tensor(False), support=positive,
            some_keyword=(o := object())
        )).infer_dict == {
            'init': s.init, 'mask': s.mask, 'support': s.support,
            'some_keyword': o
        }

    def test_distribution(self):
        with raises(TypeError, match='object is not callable'):
            _ = Sampler(d=None).distribution

        assert isinstance(Sampler(Normal(0, 1)).distribution, Normal)
        assert isinstance(Sampler(lambda: Normal(0, 1)).distribution, Normal)
        assert isinstance(Sampler(lambda: (lambda: Normal(0, 1))).distribution, Normal)

    @staticmethod
    def _test_sampler(s: Sampler, batch_shape, event_shape):
        node = pyro.poutine.trace(s.set_name('a')).get_trace().nodes['a']
        assert node['fn'].batch_shape == batch_shape
        assert node['fn'].event_shape == event_shape
        assert node['value'].shape == batch_shape + event_shape
        assert node['infer'] == dict(
            (key, val)
            for key in ('init', 'mask', 'support')
            for val in [getattr(s, key)]
            if val is not Sentinel.skip
        )

    def test_to_event_None(self):
        self._test_sampler(Sampler(
            d=Normal(0, 1).expand_by(torch.Size((5, 6))).to_event(1),
            expand_by=(3, 4)
        ), batch_shape=(), event_shape=(3, 4, 5, 6))

    def test_full(self):
        self._test_sampler(Sampler(
            d=lambda: (lambda: Normal(torch.rand(5), 1)),
            init=torch.ones((3, 4, 5, 6, 7)),
            support=positive,
            expand_by=torch.Size([3, 4]), to_event=2, indep=torch.Size([6, 7])
        ), batch_shape=(3,), event_shape=(4, 5, 6, 7))


def test_deterministic():
    node = (trace := pyro.poutine.trace(
        Deterministic(v := torch.ones(5, 6), name='a', event_dim=1)
    ).get_trace()).nodes['a']

    trace.compute_log_prob()
    assert node['value'] is v
    assert trace.log_prob_sum() == 0
    assert node['log_prob'].shape == (5,)


def test_factor():
    node = (trace := pyro.poutine.trace(
        Factor(v := torch.ones(5, 6), name='a')
    ).get_trace()).nodes['a']

    trace.compute_log_prob()
    assert pyro.poutine.util.site_is_factor(node)
    assert (node['log_prob'] == v).all()
