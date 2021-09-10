Templating
==========
.. highlight:: yaml

|Clipppy| includes rudimentary templating functionality built on top of `string.Template`. Placeholders are valid Python identifiers introduced by a ``$`` [#doubledollar]_ and optionally delimited by braces [#patterns]_\ : ``$_var123``, ``${var}``. Replacement strings are given as keywords to `ClipppyYAML.load`, `load_config`, and `loads`, e.g.

.. code-block:: python3

    >>> loads('[$var, ${var}, $var_123]', var='rep', var_123='rep_123')
    ['rep', 'rep', 'rep_123']

Template substitution is activated anytime "excess" keyword arguments are given, or when the ``force_templating`` argument to `ClipppyYAML.load` / `load_config` / `loads` is `True` (it is by default).

More usefully, templates can have defaults, specified as::

    ${var = default text }  →  default text

In this case the ``{}`` are mandatory, and surrounding whitespace is stripped. The default can be enclosed in parentheses (necessary when it contains a closing brace or to preserve surrounding whitespace)::

    ${var = (␣{text}␣␣) }  →  ␣{text}␣␣

Inside the default text a backslash and a closing parenthesis are escaped::

    ${var = f(x\)}     →  f(x)
    ${var = f(x)}      →  f(x)  # OK because doesn't start with "("
    ${var = (f(x\))}   →  f(x)  # here, though, it's necessary
    ${var = \\text\\}  →  \text\

Defaults apply only to specific instances of the template, i.e. they are not associated with the name of the placeholder. Thus, one can have different defaults in different places:

.. code-block:: python3

    >>> loads('${var=a} $var ${var=b}')
    'a $var b'
    >>> loads('${var=a} $var ${var=b}', var=7)
    '7 7 7'

Notice how in the first case the middle instance, which has no default, is left alone, and that one can pass non-string values (they are formatted with `str`).

Using templates to give the values of YAML "variables" allows spelling the default out only once::

    defs:
        - &a ${a=26}
        - &b ${b=42}
    # later, use the variables:
    a nice number: *a
    the answer: *b


.. rubric:: Footnotes
.. [#doubledollar] A literal ``$`` has to be doubled, i.e. ``$$var`` → ``$var``, when template substitution is on.
.. [#patterns] The formal pattern is :regex:`\$(?P<brace>{)?[_a-z][_a-z0-9]*(?(brace)}|)` (`regex101 <https://regex101.com/r/vmcD3b>`__) or, allowing for defaults (`regex101 <https://regex101.com/r/BT2NQw>`__):

    .. code-block:: regex
        :class: wrapped

        \$(?P<brace>{)?(?P<named>[_a-z][_a-z0-9]*)(?:|\s*=\s*(?P<paren>\()?(?P<default>(?:[^\\]|\\(?:\\|\)))*?)(?(paren)\)|)\s*)(?(brace)}|)