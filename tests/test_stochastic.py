from math import exp, pi, sqrt
from operator import itemgetter
from pyro.distributions import Normal

from clipppy.stochastic import Param, PseudoSampler, Sampler, StochasticSpecs
from clipppy.utils import Sentinel


class TestStochastic:
    def test_basic(self):
        res = dict(StochasticSpecs(dict(a=Param(), b=(n := Normal(0, 1))), f=(f := lambda: None), o=(o := object())))
        assert isinstance(res['a'], Param) and res['a'].name == 'a'
        assert isinstance(res['b'], Sampler) and res['b'].name == 'b' and res['b'].d is n
        assert isinstance(res['f'], PseudoSampler) and res['f'].func_or_val is f
        assert res['o'] is o

    def test_merge(self):
        assert dict(StochasticSpecs({
            Sentinel.merge: (m := dict(c=42, d=26))
        })) == dict(StochasticSpecs({
            Sentinel.merge: (lambda: m)
        })) == m

    def test_extract(self):
        assert dict(StochasticSpecs({
            ('e', 'pi'): (m := dict(**(e := dict(e=exp(1), pi=pi)), phi=(sqrt(5)+1)/2))
        })) == dict(StochasticSpecs({
            ('e', 'pi'): (lambda: m)
        })) == e

    def test_all_together(self):
        res = dict(StochasticSpecs({
            'a': Param(), 'b': Normal(0, 1),
            Sentinel.merge: (m := dict(c=42, d=26)),
            ('e', 'pi'): dict(**(e := dict(e=exp(1), pi=pi)), phi=(sqrt(5)+1)/2)
        }, f=(f := lambda: None), o=(o := object())).items())

        assert res.keys() == {*'abcdefo', 'pi'}
        assert res['a'].name == 'a' and res['b'].name == 'b'
        assert dict(zip(keys := (*'cde', 'pi'), itemgetter(*keys)(res))) == dict(**m, **e)
        assert res['f'].func_or_val is f
        assert res['o'] is o
