import operator as op
from collections import namedtuple

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from pytest import mark

from clipppy import loads


@mark.parametrize('name, res', {
    '==': op.eq, 'ne': op.ne, 'lt': op.lt, 'le': op.le, 'gt': op.gt, 'ge': op.ge,
    '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv,
    '@': op.matmul, '**': op.pow
}.items())
@given(*2*(arrays(float, (3, 3), elements=st.floats(
    min_value=1, allow_nan=False, allow_infinity=False)),))
def test_binop(name, res, arg1, arg2):
    assert (loads(f'''
        a: &a !py:arg1
        b: &b !py:arg2
        res: !{name} [*a, *b]
    ''')['res'] == res(arg1, arg2)).all()


# noinspection PyUnusedLocal
def test_getitem():
    mapping = {'a': 1, 'b': (obj := object())}
    seq = (1, obj)
    assert loads('![] [!py:mapping , b]') is loads('![] [!py:seq , 1]') is obj


def test_getattr():
    o = namedtuple('example', ('a',))(object())
    assert loads('!. [!py:o , a]') is o.a


def test_call():
    f = lambda arg, *, kw: (arg, kw)
    a, b = object(), object()
    assert loads('!() {/: !py:f , /: !py:a , kw: !py:b }') == (a, b)


@given(*3*(st.one_of(st.integers(min_value=0), st.none()),))
def test_slice(a, b, c):
    a_, b_, c_ = map((lambda _: 'null' if _ is None else _), (a, b, c))

    assert loads(f'!: [{a_}]') == slice(a)
    assert loads(f'!: [{a_}, {b_}]') == slice(a, b)
    assert loads(f'!: [{a_}, {b_}, {c_}]') == slice(a, b, c)

