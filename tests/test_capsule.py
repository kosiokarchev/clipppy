from clipppy.stochastic.capsule import AllEncapsulator, Capsule, Encapsulator
from clipppy.stochastic.stochastic import Stochastic


class Weakrefable(object):
    pass


def test_capsule_basic():
    c = Capsule(float('inf'))
    c.value = o = object()
    assert c.value is c() is o


def test_lifetime():
    assert Capsule().remaining == 0

    c = Capsule.init(o := Weakrefable(), lifetime=2)
    assert c.remaining == c.lifetime == 2
    assert c.value is c() is o
    assert c.remaining == 0


def test_as_stochastic():
    assert Stochastic(lambda **kwargs: kwargs, specs={
        ('a', 'b'): Capsule.init({
            'a': (oa := object()), 'b': (ob := object()), 'c': object()}),
        'c': Capsule.init(oc := object())
    })() == {'a': oa, 'b': ob, 'c': oc}


def test_all_encapsulator():
    o = {'a': object(), 'b': object(), 'c': object()}
    aec = AllEncapsulator(lambda: o, Capsule(), Capsule(), Capsule(), b=Capsule(), c=Capsule())
    assert aec() is aec.capsule() is o
    assert tuple(map(Capsule.__call__, aec.capsule_args)) == tuple('ab')
    assert aec.capsule_kwargs.keys() == set('bc')
    assert aec.capsule_kwargs['b']() is o['b'] and aec.capsule_kwargs['c']() is o['c']


def test_encapsulator():
    o = {'a': object(), 'b': object(), 'c': object()}
    ec = Encapsulator(lambda: o, Capsule(), Capsule(), b=Capsule(), c=Capsule())
    assert ec() is o
    assert tuple(map(Capsule.__call__, ec.capsule_args)) == tuple('ab')
    assert ec.capsule_kwargs.keys() == set('bc')
    assert ec.capsule_kwargs['b']() is o['b'] and ec.capsule_kwargs['c']() is o['c']
