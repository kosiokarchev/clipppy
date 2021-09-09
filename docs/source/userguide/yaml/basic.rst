Basic YAML
==========

YAML is [#aint]_ a markup language that extends JSON. Hence, any valid JSON, like :json:`{"answer": 42, "foo": [3.14, "euler's number", {"question": null, "whatever": {"maybe": true, "but actually": false}}]}` is valid YAML. This basic variant (which is almost directly usable as a Python literal) allows the description of arbitrary primitive objects: numbers, strings, arrays (Python lists), and dictionaries. YAML also allows a modified syntax, where one is permitted to

- ditch the quotation marks since everything that doesn't look like a number (and isn't a literal like :yaml:`true`, :yaml:`false`, :yaml:`null`) is interpreted as a string;
- ditch the brackets as long as one uses indentation: everything more indented with respect to the parent by the same amount is on the same nesting level;
- use a bullet-point-like style for lists (using :yaml:`-`) instead of square brackets.

Thus, in YAML the above example may be rewritten as ::

    answer: 42
    foo:
      - 3.14
      - euler's number
      - question: null
        whatever: {maybe: true, but actually: false}

which is, arguably, way more pleasant to look at. [#fretnot]_ Note that one can still use (even partially well-formatted) JSON for any node.

.. note:: YAML also permits using lists (will be converted to tuples) as keys in a dictionary::

        [a string, 26]: value

    However, the usual hashability rules of Python apply, so dictionaries are not allowed inside keys. Also, since non-string keys are not used in function definitions, this feature is discouraged in |Clipppy|, and no guarantees are made that it will indeed be allowed forever.

.. rubric:: Footnotes
.. [#aint] according to its name, though, it ain't
.. [#fretnot] Fret not! It will soon become much messier!

Advanced YAML [#intermediate]_
==============================

On top of syntactic sugar, YAML comes with some useful additional features. One of them is the ability to name and subsequently reference nodes. The syntax is inspired by C's pointers::

    a: &name {key1: value1, key2: value2}
    b: *name

Using :yaml:`&name` defines the "variable", and :yaml:`*name` "dereferences" it [#pointers]_. The pointer language is accurate here since in the parsed object, the two nodes will be converted to *references to the same object*, so :python:`parsed['a'] is parsed['b']` will evaluate to `True` in Python. Since this is a standard feature of `ruamel.yaml`_, |Clipppy|'s machinery is bypassed when dereferencing, which might be surprising to someone who uses YAML references as a way to avoid duplicating code and doesn't really mean to have the same object.

.. rubric:: Footnotes
.. [#intermediate] well, intermediate, really
.. [#pointers] The correct terms in YAML-speak are "anchor" and "alias".

.. include:: /userguide/yaml/tags.rst

Scopes
------

The directives `\!py` and `\!eval` advertise giving you access to arbitrary Python (objects) from inside the YAML configuration and therefore need to resolve variable names. The scope in which this is done is kept in |scopevar| [#scope]_. By default |Clipppy| makes every\ |citation needed| effort to simulate the scoping "rule" of `eval`/`exec`, i.e. to "execute" the YAML in the local scope from which `loads`/`clipppy.load_config` or `ClipppyYAML.load` is called:

.. code-block:: python3

    >>> a = 'spam, baked beans, and spam'
    >>> clipppy.loads('!py:str.replace [!py:a , baked beans, spam]')
    'spam, spam, and spam'

(Note the space here, since a) we don't want to call ``a``, and b) a space is required after every tag in YAML.)

To achieve this, every invokation of `ClipppyYAML.load` by default collects the locals, globals, and builtins from the appropriate `frame <types.FrameType>` and saves them to |scopevar|. The scope may then be *updated* by `\!import` directives, and these updates **will leak** to the caller. This is probably best illustrated with an explicitly given scope:

.. code-block:: python3

    >>> scope = {}
    >>> clipppy.loads('!import numpy as np', scope=scope)      # None
    >>> scope
    {'np': <module 'numpy' from '.../numpy/__init__.py'>}
    >>> clipppy.loads('!import jax.numpy as np', scope=scope)  # None
    >>> scope
    {'np': <module 'jax.numpy' from '.../jax/numpy/__init__.py'>}
    >>> 'jax.numpy' in sys.modules
    True

but the same thing happens when using the default "current" scope:

    >>> clipppy.loads('!import torch')  # uses the current scope
    >>> torch
    <module 'torch' from ...>

On top of that scope, `ClipppyYAML` installs a custom |builtinsvar| that consists of the usual `__builtins__ <python:builtins>` and the global scope of `clipppy.yaml`. The latter is kept for compatibility and as an easy way to get `numpy`, `torch`, and the majority of the `clipppy` API registered, even though the "full" API is then explicitly registered in this |builtinsvar| scope.

.. note:: If invoked from within YAML, e.g. via :yaml:`!py:locals []` or :yaml:`!py:globals []`, the built-in `locals` and `globals` functions return the respective scopes for some function inside `clipppy.yaml` instead of something more meaningful [#safemode]_. The way to get at the "correct" scope, which `\!import` imports in, is via `eval`-uating `locals`/`globals` as a Python call: :yaml:`!eval locals()`, which will return |scopevar| as, currently, a `~collections.ChainMap`. Remember, though, that `\!py` operations essentially transpire in this scope anyway.

.. rubric:: Footnotes
.. [#scope] This attribute is unconditionally overwritten on each `~ClipppyYAML.load`, so setting it directly will not have an effect on YAML loading. What it is set to, though, is controlled by the :arg:`scope` function parameter, which is your chance of controlling the YAML "globals" scope'; especially, if you want to "hide" the caller scope from the YAML for some reason (speed?), pass an empty dictionary.
.. [#safemode] This might point you to why loading YAML is considered "unsafe" and why `ruamel.yaml`_ operates in a "safe" mode, turning which off is the first order of business for `ClipppyYAML`.