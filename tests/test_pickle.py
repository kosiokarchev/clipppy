import pickle

from pytest import mark

from clipppy.stochastic import Stochastic
from clipppy.utils import Sentinel

o = object()


def func(arg):
    return o, arg


def test_pickle_stochastic():
    original = Stochastic(func, {'arg': 7})
    pickled = pickle.dumps(original)
    rec: Stochastic = pickle.loads(pickled)

    assert original._wrapped_obj.func is rec._wrapped_obj.func
    assert original.stochastic_specs.specs == rec.stochastic_specs.specs
    assert original() == rec()


@mark.parametrize('s', tuple(Sentinel))
def test_pickle_sentinel(s):
    assert pickle.loads(pickle.dumps(s)) is s
