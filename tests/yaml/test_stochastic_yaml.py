import pyro, pyro.distributions
import torch
from torch.distributions import biject_to
from torch.distributions.constraints import positive

from clipppy import loads
from clipppy.stochastic.stochastic import Stochastic
from clipppy.stochastic.capsule import AllEncapsulator, Encapsulator
from clipppy.stochastic.sampler import Sampler, Context, Deterministic, Effect, Factor, Param, PseudoSampler, UnbindEffect


def test_registered():
    null_params = {
        Param: '', Context: 'null, null', Effect: '!py:None , null',
        AllEncapsulator: '!eval "lambda: None", null',
        Encapsulator: '!eval "lambda: None"',
        Stochastic: '!eval "lambda: None", {}',
    }
    named = (Param, Sampler, Deterministic, Factor)

    for cls in (Param, Sampler, Deterministic, Factor,
                PseudoSampler, Context, Effect, UnbindEffect,
                AllEncapsulator, Encapsulator, Stochastic):
        arg = null_params.get(cls, 'null')
        assert isinstance(loads(f'!{cls.__name__} [{arg}]'), cls)
        if cls in named:
            assert loads(f'!{cls.__name__}:name [{arg}]').name == 'name'

    assert loads(f'!Stochastic:name [{null_params[Stochastic]}]').stochastic_name == 'name'


class TestStochasticSpecs:
    def test_basic(self):
        # This also tests that StochasticSpecs can be instantiated as if
        # StochasticSpecs(**specs) when in reality it is StochasticSpecs(specs)
        p, n = Param(), pyro.distributions.Normal(0, 1)
        res = dict(loads('!py:StochasticSpecs {a: !py:p , b: !py:n }'))
        assert res['a'] is p and res['b'].d is n

    def test_merge(self):
        assert dict(loads('!py:StochasticSpecs {<<: {c: 42, d: 26}}')) == {'c': 42, 'd': 26}

    def test_extract(self):
        assert dict(loads('!py:StochasticSpecs {[e, pi]: {e: 2.718, pi: 3.142, phi: 1.618}}')) == {'e': 2.718, 'pi': 3.142}

    def test_all_together(self):
        p, s = Param(), Sampler(pyro.distributions.Normal(0, 1))
        assert dict(loads('''!py:StochasticSpecs
            a: !py:p
            b: !py:s
            <<: {c: 42, d: 26}
            [e, pi]: {e: 2.718, pi: 3.142, phi: 1.618}
            <<: !eval "lambda: {'g': 9.81, 'h': 6.626e-34}"
            [i, j]: !eval "lambda: {'i': 0, 'j': 1j, 'k': 2}"
        ''')) == {'a': p, 'b': s, 'c': 42, 'd': 26, 'e': 2.718, 'pi': 3.142,
                  'g': 9.81, 'h': 6.626e-34, 'i': 0, 'j': 1j}

    def test_capsules(self):
        def reveal_kwargs(**kwargs):
            return kwargs

        exec_counter = [0]

        def get_stuff():
            exec_counter[0] += 1
            return {'the answer': 42, 'a number': 26}

        s = loads('''!Stochastic:name
            - !py:reveal_kwargs
            - a: !AllEncapsulator
                /: !py:get_stuff
                /: &b
                <: [&c, &d]
                <<: {the answer: &e}
              b: *b
              c: *c
              d: *d
              e: *e
              [a number]: *b
        ''')
        res = s()

        assert exec_counter[0] == 1
        assert res['a'] is res['b']
        assert res['c'] == 'the answer' and res['d'] == 'a number'
        assert res['e'] == 42 and res['a number'] == 26


def test_pseudo():
    o = object()
    fo = lambda: o

    assert loads('!PseudoSampler [!py:fo ]')() is o
    assert loads('!PseudoSampler {/: !py:fo , call: true}')() is o
    assert loads('!PseudoSampler {/: !py:fo , call: !py:Sentinel.call }')() is o
    assert loads('!PseudoSampler [!py:o ]')() is o
    assert loads('!PseudoSampler {/: !py:fo , call: false}')() is fo
    assert loads('!PseudoSampler {/: !py:fo , call: !py:Sentinel.no_call }')() is fo


def test_context():
    node = pyro.poutine.trace(
        loads('!Context [!py:pyro.plate [plate, 100],'
              '          !Sampler:a [!py:pyro.distributions.Normal [0, 1]]]')
    ).get_trace().nodes['a']
    assert node['value'].shape == (100,)
    assert node['cond_indep_stack'][0].name == 'plate'


class TestEffects:
    def test_unbind_effect(self):
        assert all(a.shape == (5,) for a in loads('!UnbindEffect [!py:torch.rand [5, 6]]')())
        assert all(a.shape == (6,) for a in loads('!UnbindEffect {/: !py:torch.rand [5, 6], dim: 0}')())


def test_param():
    pyro.clear_param_store()
    p = loads('''!Param:a
        init: !py:torch.rand [10]
        support: !py:positive
        event_dim: 1
    ''')
    assert ((v := p()) == pyro.param('a')).all()
    assert (biject_to(positive)(v.unconstrained()) == v).all()


class TestSampler:
    def test_name(self):
        assert loads('!Sampler [null]').name is None
        assert loads('!Sampler:a [null]').name == 'a'
        assert loads('!Sampler {/: null, name: a}').name == 'a'

    def test_infer_dict(self):
        o = object()
        assert (s := loads('''!Sampler
            /: null,
            init: !py:torch.tensor [42]
            mask: !py:torch.tensor [false]
            support: !py:positive
            some_keyword: !py:o
        ''')).infer_dict == {
            'init': s.init, 'mask': s.mask, 'support': s.support,
            'some_keyword': o
        }


def test_deterministic():
    v = torch.ones(5, 6)
    node = (trace := pyro.poutine.trace(loads(
        r'!Deterministic:a {/: !py:v , event_dim: 1}'
    )).get_trace()).nodes['a']

    trace.compute_log_prob()
    assert node['value'] is v
    assert trace.log_prob_sum() == 0
    assert node['log_prob'].shape == (5,)
