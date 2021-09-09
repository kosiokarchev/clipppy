From Node to Signature
======================
.. highlight:: python3

Magic Keys
----------
There are only three "magic keys". Since YAML does not allow mixing sequence and mapping nodes, while in Python this is common practice, and also to cover the case of :ref:`positional-only parameters <python:positional-only_parameter>`, |CLipppy| needs a positional argument indicator key. Furthermore, since it is common to want to expand some generated parameter or maybe use the same object as a monolithic sequence in one place and as individual items in another [#forget]_, |Clipppy| defines positional and keyword expansion "operators" corresponding to the Python :ref:`parameter expansion syntax <python:calls>` ``*``/``**``.

.. glossary::

    ``/``
        Use the value as a positional argument. Can be used at any point (even after keywords, contrary to the Python grammar).

    ``<``
        Expand the value into positional arguments. A simple use case would be some ``xy`` coordinates as an :math:`N \times 2` array that need to be expanded into two arrays of length :math:`N`:

        .. code-block:: yaml

            - &pts [[0, 0], [1, 1], [26, 42]]
            ...
            - !py:matplotlib.pyplot.plot
                <: !py:np.transpose [*pts]

        which corresponds to the very similar Python code

        .. code-block:: python3

            plt.plot(*np.transpose(pts))

        .. note::
            If you try this example with ``ruamel.yaml<=0.17.4`` (or maybe even higher), this **will** (may) **not work!** The reason is that there is no (not-too-hacky) way to force depth-first construction if using an optimised C-based loader/parser/constructor, and the current implementation returns *an empty list* as the value of the referenced node when the :yaml:`!py:np.transpose`-tagged node requires it. To solve this, tag the whole document with :yaml:`!py:list` for example, which will transfer control to `ClipppyYAML` from the beginning (and make the document a one-element sequence as per the requirement of `list`... See, I told you: hacky!).

            This highlights a fundamental design choice of |Clipppy|: in order to provide sensible insight using type hints, construction has to be depth first and recursive (hence, Python's stack depth limitation applies to |Clipppy| YAML files). In contrast, simple *collection assembly* can live with breadth-first construction and a subsequent population using further placeholders, etc.

        .. deprecated:: 0
            Initially, the key for positional expansion was ``__args``, but this should not be used anymore.

    ``<<``
        Expand the value into keyword arguments. This "merge type" is actually `present <https://yaml.org/type/merge.html>`__ in the officially recommended `YAML type system <https://yaml.org/type/>`_ [#merge]_. |Clipppy| needs to merge eagerly, though, in order to be able to tag the nodes, so this key is handled specially. Otherwise, it does what you would expect: merges the named mapping into its parent, *overwriting* any already present keys. In this regard

        .. code-block:: yaml

            !py:func {<<: *map1, <<: *map2, ...}

        behaves more like ::

            func(**{**map1, **map2, ...})

        than ::

            func(**map1, **map2, ...)

        which would throw an exception for repeated keys. The same overwriting rule applies to keys not from expanded mappings.

Magic keys can be freely mixed and matched, used multiple times, etc. The order of evaluation of the nodes/parameters follows strictly the definition order in the YAML, just as it follows the definition order in a Python call (important for side effects and defining anchors). Here's an example:

.. code-block:: yaml

    !py:f
    <: [!eval 22/7]
    euler: 2.72
    <: [0, 1, 1, 2, 3, 5, 8]
    euler: 2.71828
    <<: {euler: !py:math.exp [1],
         pi: !py:math.pi , phi: !py:mpmath.phi }  # spaces!

may be used with a signature like ::

    def f(not_pi, *fibonacci, euler, **exact): ...

and will result in a `locals` ::

    {'not_pi': 3.142857142857143,
     'euler': 2.718281828459045,
     'fibonacci': (0, 1, 1, 2, 3, 5, 8),
     'exact': {'pi': 3.141592653589793,
               'phi': <Golden ratio phi: 1.61803~>}}

.. rubric:: Footnotes
.. [#merge] But... it isn't really a type, is it? It's a procedural directive, mandating the *merge* of some mappings, which is an *operation*!
.. [#forget] or simply to forget the name of some commonly-used first parameter, like in the :ref:`example <sorted-example>` with `sorted` above. In that case, you'll need to wrap it in ``[]``, of course.


The Power of Type Hints
-----------------------

`Type hints <python:typing>` in Python are the best! [#pep563]_ They are completely ignored at runtime, so they don't limit you in any way, but are still tremendously helpful in static analysis and allow IDEs to spot errors in your code before you run it. They help clarify the meaning of parameters and properties and contribute to automatic documentation generation. Even though the language ignores type hints, they are not completely "lost" as are the types of compiled languages: "annotations" can be freely examined by the program using the builtin `typing` and `inspect` modules. Basically, they are free information that the software designer gives to the program without any obligation. As such, type hints are often the basis of "smart" functionality, such as in the `dataclasses` modules. And in |Clipppy|, which tries to be smart and save you some typing in YAML if you have gone through the trouble of writing properly annotated Python code.

|Clipppy| needs to invoke Python functions with arguments coming from YAML in order to construct complex data structures beyond simple containers (sequences and mappings). Sometimes the inputs are themselves complex structures, and so the YAML parser needs to be informed further of the way to form them from simpler data, and so on. However, the original function knows what data to expect, and the constructors of complex structures know what primitives they need, or at lest the programmer who wrote them does. Thus, if they provided this information as type hints, |Clipppy| can try to automatically determine the processing needed in the middle between primitives and the final call signature.

Take the following typical |Clipppy| configuration as example:

.. code-block:: yaml

    guide:
        - cls: MultivariateNormalSamplingGroup
          name: main
          match: main/.*
        - cls: DiagonalNormalSamplingGroup
          name: others

To an outside observer this is just a one-key mapping, and the one value is a list of two further mappings with some strings. No tags or further information provided. However, as we said, |Clipppy| can automatically assume that this whole YAML represents a `Clipppy` object, and so automatically tag it [#interpretAsClipppy]_ with :yaml:`!py:Clipppy`. The node, thus, represents a call to the constructor of `Clipppy` with an argument :arg:`guide`, so |Clipppy| `inspect`\ s it for further information. In an ideal world, such as the one we live in, the :arg:`guide` parameter would be tagged with `Guide` so that the parser can tag it with :yaml:`!py:clipppy.guide.guide.Guide` (it's a mouthful, but that's qualified names for you; also, that's why we want automation, right?). Next, the constructor for `Guide` reads ::

    def __init__(self, *specs: GroupSpec, model=None, name=''): ...

so the parser expands the sequence node into this signature and realises than both elements should be instances of `GroupSpec`, whose constructor is

.. parsed-literal::

    def __init__(
        self,
        cls: `~typing.Type`\ [`SamplingGroup`] = `DeltaSamplingGroup`,
        match: `~typing.Union`\ [`str`, `re.Pattern <python:re-objects>`] = _allmatch,
        exclude: `~typing.Union`\ [`str`, `re.Pattern <python:re-objects>`] = _nomatch,
        name='', \*args, \*\*kwargs): ...

Here, even though ``name`` is not annotated, |Clipppy| will consider the type of the default value in line with most type checkers. However, a `str` is not particularly interesting since scalar nodes are by default strings. The :arg:`match` is a `~typing.Union` for convenience and is explicitly converted to a `re.Pattern <python:re-objects>` in the body of the function. Sadly, |Clipppy| connot handle `~typing.Union`\ s yet, so it leaves the :arg:`match` node alone [#regex]_. Finally, for the :arg:`cls` parameter, meant to indicate the subtype of `SamplingGroup` to use, |Clipppy| assumes that the node is a *name* of a class / Python object to pass. The node is then tagged with :yaml:`!py:VALUE`, where ``VALUE`` is the original content [#typechecks]_. |Clipppy| does that for all `~typing.Type` or `typing.Callable`\ \|\ `collections.abc.Callable` annotations, so if you want to pass something else than a name, you should put an explicit annotation.

Depending on `ClipppyConstructor`\ ``.``\ `~TaggerMixin.strict_node_type`, which is `True` by default, |Clipppy| enforces the types of nodes versus what it expects from an annotation: that callable / string parameters are represented as scalar nodes and that builtin sequences / mappings are, respectively, sequences / mappings.

Finally, the original YAML is perceived as

.. code-block:: yaml

    !py:Clipppy
    guide: !py:clipppy.guide.guide.Guide
        - !py:clipppy.guide.sampling_group.SamplingGroup
            cls: !py:MultivariateNormalSamplingGroup
            name: main
            match: main/.*
        - !py:clipppy.guide.sampling_group.SamplingGroup
            cls: !py:DiagonalNormalSamplingGroup
            name: others

.. rubric:: Footnotes
.. [#pep563] But they're soon getting worse (:pep:`563`)... :/
.. [#interpretAsClipppy] This only applies to loading with ``interpret_as_Clipppy``, as discussed above. Note that |Clipppy| will *never* interfere with your code if you're explicit and do put tags in, unless they are the standard ones ``<tag:yaml.org,2002:str>``, ``<...:seq>``, ``<...:map>``, which are actually auto-assigned based on the node type.
.. [#regex] Even if the annotation were a plain `re.Pattern <python:re-objects>`, it wouldn't work directly. |Clipppy| may be smart, but how is it to know that the constructor raises a ``TypeError: cannot create 're.Pattern' instances`` when called directly, or that its signature checks out as ``()``, i.e. nothing?! Maybe the developer knows that, though, and also that `Pattern <python:re-objects>`\ s are constructed via `re.compile`. They can then help |Clipppy| by registering a type-to-tag mapping in `ClipppyConstructor`\ ``.``\ `~TaggerMixin.type_to_tag` as ::

        ClipppyConstructor.type_to_tag[re.Pattern] = '!py:re.compile'

    to replace the default ``cls -> '!py:{cls.__module__}.{cls.__name__}'``. Then a function like :python:`f(a: re.Pattern)` can be safely "called" as ``!py:f [``\ `(meta-)*regex golf <https://xkcd.com/1313/>`_\ ``]`` and will be passed :python:`re.compile('(meta-)*regex golf')`.

.. [#typechecks] For now no checks for inheritance / signature constraints or types of container elements are performed by |Clipppy|, so this has to be handled in user code.

.. YAML is a data serialisation language: it doesn't have the concept of functions or change. And why would it, data and the relationships between it are usually static, right? In contrast, |CLipppy| is a probabilistic "framework" whose whole point is to facilitate working with variational and sampling methods. The "data" those work with are certainly the opposite of static, and that's why a "real" programming language beyond YAML is needed to perform inference. What is invariant, though, are the *relationships* between different components of a model, and it is those that |Clipppy| outsources to its YAML configuration.

.. YAML is usually used to serialise static data and express the relationships between its different pieces. In contrast, |Clipppy| deals with the *creation* and subsequent *functioning* of objects. Since YAML has no clue about change, |Clipppy| requires that it be described in a "real" programming language, while outsourcing to YAML only the *structure* of models, which is usually invariant. Whereas in YAML the creation *process* is simply an *assembly* of primitive data into containers, |Clipppy| regards it in its full complexity, allowing arbitrary transformations between input "primitive" data (and subsequently of more complicated structures) and more complicated structures, otherwise known as *functions*.