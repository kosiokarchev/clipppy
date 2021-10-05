import collections.abc
import io
import re
import typing
from itertools import chain, repeat
from typing import Callable, Iterable, Mapping, Optional, Type, Union

import torch
from pytest import mark, raises, warns
from torch import Tensor

from clipppy import ClipppyYAML, loads
from clipppy.yaml.constructor import ClipppyConstructor
from clipppy.yaml.tagger import NodeTypeMismatchError, NodeTypeMismatchWarning


def func_of_tensors(*ts: Tensor):
    return ts


def func_of_regex_and_tensor(pattern: re.Pattern, t=torch.ones(42)):
    return pattern, t


def func_of_union(arg: Union[Tensor, re.Pattern]):
    return arg


def func_of_optional(t: Optional[Tensor] = None):
    return t


def func_of_nontyped(a: 3.14, b=None):
    return a, b


def func_of_generic_of_tensors(
        args: Iterable[Tensor], kwargs: Mapping[re.Pattern, Tensor]):
    return args, kwargs


def test_basic_type_hints():
    ClipppyConstructor.type_to_tag[re.Pattern] = '!py:re.compile'

    assert all(isinstance(t, Tensor) for t in loads(
        '!py:func_of_tensors [42, 26]'))
    assert tuple(map(type, loads(
        '!py:func_of_regex_and_tensor [(meta-)*regex golf, 42]'
    ))) == (re.Pattern, Tensor)
    assert all(isinstance(t, Tensor) for t in loads(
        '!py:func_of_regex_and_tensor [!py:Tensor 42, 26]'))
    assert isinstance(loads('!py:func_of_union [random]'), str)
    assert isinstance(loads('!py:func_of_optional [42]'), Tensor)
    assert loads('!py:func_of_nontyped [6.28, abc]') == (6.28, 'abc')

    res = loads('''!py:func_of_generic_of_tensors
        - [42, 26]
        - {(ba)+: 42, (meta-)*regex golf: 26}
    ''')
    assert all(isinstance(r, Tensor) for r in res[0])
    assert all(isinstance(p, re.Pattern) for p in res[1].keys())
    assert all(isinstance(r, Tensor) for r in res[1].values())


class Strange:
    def __init__(self, a: str):
        raise TypeError('strange')


def test_invalid():
    def f1(a, b): pass
    with raises(TypeError, match='Cannot bind'):
        loads('!py:f1 [1, 2, 3]')
    with raises(TypeError, match='Cannot bind'):
        loads('!py:f1 {c: 1, d: 2, e: 3}')

    def f2(*args): return args
    assert loads('!py:f2 [1, 2, 3]') == (1, 2, 3)

    def f3(a: Strange): ...
    with raises(TypeError, match='strange'):
        loads('!py:f3 [test]')


@mark.parametrize('hint, argtype', list(chain(*(zip(val, repeat(key)) for key, val in {
    'a': (typing.Callable, collections.abc.Callable, str),
    '[]': (list, typing.List, typing.Iterable, collections.abc.Iterable,
           typing.List[str], typing.Iterable[str], collections.abc.Iterable[str]),
    '{}': (dict, typing.Dict, typing.Mapping, collections.abc.Mapping,
           typing.Mapping[str, str], typing.Mapping[str, str], collections.abc.Mapping[str, str])
}.items()))))
def test_target_nodetype(hint, argtype):
    def func(a: hint): ...

    for node in {'a', '[]', '{}'} - {argtype}:
        s = f'!py:func {{ a: {node} }}'

        with raises(NodeTypeMismatchError):
            loads(s)

        if hint not in (typing.Callable, collections.abc.Callable):
            yaml = ClipppyYAML()
            yaml.constructor.strict_node_type = False
            with warns(NodeTypeMismatchWarning):
                yaml.load(io.StringIO(s))


def test_name_for_callable():
    # if hint is Callable or a Type, convert string to !py:...
    def func(f: Callable, g: Type[Iterable], h=tuple):
        return f, g, h

    assert loads(
        '!py:func [func_of_tensors, list, dict ]'
    ) == (func_of_tensors, list, dict)


def test_auto_provide_if_anchor():
    # if a hinted node is empty but anchored, put the hint and call it
    def func(t: object):
        return t

    res = loads('[!py:func [ &name ], *name]')
    assert type(res[0]) is object and res[0] is res[1]

