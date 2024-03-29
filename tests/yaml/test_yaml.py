import io
import operator
import os
import sys
from operator import attrgetter, is_, itemgetter
from pathlib import Path
from warnings import catch_warnings, filterwarnings

import numpy as np
import torch
from pytest import fixture, mark, raises

from clipppy import Clipppy, load, load_config, loads
from clipppy.yaml import cwd
from clipppy.yaml.tagger import NodeTypeMismatchError


SOME_GLOBAL_VARIABLE = 42


def reveal_args(*args, **kwargs):
    return args, kwargs


def test_basic():
    assert is_(*itemgetter('a', 'b')(loads('''
        a: &name {key1: value1, key2: value2}
        b: *name
    ''')))


def test_templating():
    assert loads('${var=abc}', other='def') == 'abc'
    assert loads('${var=abc}', other='def', force_templating=False) == 'abc'
    assert loads('${var=abc}', force_templating=False) == '${var=abc}'
    assert loads('${var}') == '${var}'
    assert loads('$var $_var', var='abc', _var='def') == 'abc def'
    assert loads('${var}', var='abc') == 'abc'
    assert loads('$${var=(${var=abc})}', var=123) == '${var=(123)}'

    # whitespace handling
    assert loads('"${var = abc }"') == 'abc'
    assert loads('"${var =( abc )}"') == ' abc '

    # escapes
    assert loads(r'${var=(abc{(\)})}') == r'abc{()}'
    assert loads(r'${var=\\text\\}') == '\\text\\'
    assert loads(r'${var=\text\\}') == '${var=\\text\\\\}'
    assert loads('$${var=abc}') == '${var=abc}'
    assert loads('$, $123') == '$, $123'


def test_paths(tmp_path: Path):
    curdir = os.getcwd()
    assert curdir != str(tmp_path)
    assert loads('!py:os.getcwd []', base_dir=tmp_path) == str(tmp_path)
    assert os.getcwd() == curdir


def test_py():
    SOME_LOCAL_VARIABLE = 26
    assert loads('[!py:SOME_GLOBAL_VARIABLE , !py:SOME_LOCAL_VARIABLE ]') == [SOME_GLOBAL_VARIABLE, SOME_LOCAL_VARIABLE]

    with raises(NameError, match='SOME_LOCAL_VARIABLE'):
        loads('[!py:a , !py:SOME_LOCAL_VARIABLE ]', scope={'a': 42})

    with raises(ModuleNotFoundError):
        loads('!py:nonsense_blahblah.somename')

    assert loads('!py:print ') is print
    assert loads('!py:str.join [" ", [Hooray, for, Python!]]') == str.join(' ', ['Hooray', 'for', 'Python!'])
    assert loads('''
        !py:sorted
            <: [[[a, 42], [c, 26], [b, 13]]]
            key: !py:operator.itemgetter [0]
            <<: {reverse: True}
    ''') == sorted(*[[['a', 42], ['c', 26], ['b', 13]]], key=itemgetter(0), **{'reverse': True})

    a = 'spam, baked beans, and spam'
    assert loads('!py:str.replace [!py:a , baked beans, spam]') == 'spam, spam, and spam'

    assert loads('!py:os.path.split ') is sys.modules['os.path'].split


@mark.parametrize('name, modname', (
    ('numpy', 'numpy'), ('np', 'numpy'), ('torch', 'torch'), ('op', 'operator')
))
def test_preimported_modules(name, modname):
    assert loads(f'!eval {name}') is sys.modules[modname]


@mark.parametrize('name', operator.__all__)
def test_starimport_operator(name):
    assert loads(f'!py:{name}') is getattr(operator, name)


def test_import():
    scope = {}
    assert loads('''
        _: !import
            - import torch
            - import numpy as np
            - from urllib import request as req
        __: !import os.path as ospath
        a: [!py:torch.linspace , !py:np.linspace ]
    ''', scope=scope) == {'_': None, '__': None, 'a': [sys.modules['torch'].linspace, sys.modules['numpy'].linspace]}
    assert itemgetter('torch', 'np', 'req', 'ospath')(scope) == itemgetter('torch', 'numpy', 'urllib.request', 'os.path')(sys.modules)

    scope = {}
    loads('!import [{from: urllib, import: request}, {from: urllib, import: [error, parse]}]', scope=scope)
    assert scope == dict(zip(keys := ('request', 'error', 'parse'), attrgetter(*keys)(sys.modules['urllib'])))

    with raises(SyntaxError):
        loads('!import 3+4')


def test_magic():
    assert loads('{a: 1, <<: {b: 2, c: 3}}') == {'a': 1, 'b': 2, 'c': 3}

    assert loads('''
    !py:reveal_args
      a: a
      <: [1, 2]
      b: b
      /: 3
      <<: {c: c, d: d}
      <: !py:range [4, 7]
    ''') == ((1, 2, 3, 4, 5, 6), {'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd'})

    with raises(NodeTypeMismatchError):
        loads('!py:reveal_args {<<: [1, 2, 3]}')


def test_eval():
    import math
    assert loads('!eval math.pi / 2') == math.pi / 2


class TestWithFiles:
    @fixture(scope='class', autouse=True)
    def chdir(self):
        with cwd(Path(__file__).parent):
            yield

    def test_loaders(self):
        assert (
            {'answer': 42, 'foo': [3.14, 'euler\'s number', {
                'question': None, 'whatever': {
                    'maybe': True, 'but actually': False}}]}
            == loads(open('res/basic.yaml').read())
            == load(open('res/basic.yaml'))
            == load('res/basic.yaml')
        )

        assert isinstance(load_config(io.StringIO('{}')), Clipppy)

    def test_data_loaders(self):
        assert (loads('!txt res/data.txt') == np.loadtxt('res/data.txt')).all()
        rcp = loads(r'!txt {/: res/data.txt, dtype: !py:np.float32 , unpack: true}')
        assert len(rcp) == 2 and rcp[0].dtype is np.dtype(np.float32)

        assert all((_ == __).all() for _ in loads('''
            - !npy res/data.npy
            - !npy {/: res/data.npy, allow_pickle: false}
        ''') for __ in [np.load('res/data.npy')])

        assert all((_ == __).all() for _ in loads('''
            - !npz [res/data.npz, somekey]
            - !npz {/: res/data.npz, /: somekey, allow_pickle: false}
            - !py:operator.getitem [!npz res/data.npz, somekey]
        ''') for __ in [np.load('res/data.npz')['somekey']])

        assert all((_ == __).all() for _ in loads('''
            - !pt [res/data.pt, somekey]
            - !pt {/: res/data.pt, /: somekey, map_location: cpu}
            - !py:operator.getitem [!pt res/data.pt, somekey]
        ''') for __ in [torch.load('res/data.pt')['somekey']])

    def test_trace(self):
        nodes = torch.load('res/trace.pt').nodes
        for key in 'abc':
            assert loads(f'!trace [res/trace.pt, {key}]') == nodes[key]['value']

        assert loads('!trace [res/trace.pt, [a, b]]') == {
            'a': nodes['a']['value'], 'b': nodes['b']['value']
        }

    def test_tensor(self):
        res = loads('!tensor 42')
        assert torch.is_tensor(res) and res.shape == torch.Size() and res == 42
        assert loads('!tensor [42]').shape == torch.Size()
        assert (loads('!tensor [[1, 2, 3, 4]]') == torch.tensor([1, 2, 3, 4])).all()

        ddtype = torch.get_default_dtype()

        torch.set_default_dtype(torch.float32)

        assert all(_.dtype is torch.float64 for _ in loads('''
            - !tensor {/: !npz [res/data.npz, somekey], dtype: !py:torch.float64 }
            - !tensor:float64 [!npz [res/data.npz, somekey]]
        '''))
        with catch_warnings():
            filterwarnings('ignore', message='To copy construct from a tensor', category=UserWarning)
            assert loads('!tensor:default [!tensor:float64 42]').dtype is torch.get_default_dtype()

        torch.set_default_dtype(ddtype)

        with raises(ValueError, match=r'not a valid torch\.dtype'):
            loads('!tensor:gibberish 42')
