Tags
----

Secondly, YAML introduces the idea of *tags* (and |Clipppy| takes it maybe a bit too close to the heart). They are identified by a prefixed ``!``, as in ``!tag``, and, as in any markup language, define the type of the node/element. They are metadata and denote a step of postprocessing the primitive values found in the node contents.

.. note:: The specification goes on and on about tags and prefixes and namespaces and URIs and local and global and what-not tags... Truth is, every node must have a tag (be it an empty one), but end users usually don't with bother them since most tags are assigned automatically by the parser based on whether a node is a number, a string, a literal, an array, or a dictionary. |Clipppy| adds specific tags for specific object types and uses other tags to enable more complex behaviour presented below. Tags are also assigned (experimentally) based on type annotations as described in `The Power of Type Hints`_.

.. glossary::

    ``!py``
        Used as ``!py:NAME``, this tag allows access to arbitrary objects. The name can be any qualified name (non strict sense), i.e. any importable module, or nothing, followed by a sequence of attribute accesses. Maybe some examples will clarify this:

        - a name in the local scope: ``print``,
        - ``a_local_var.s_attribute.and_more``,
        - a series of modules and a name: ``astropy.units.astrophys.attoparsec``,
        - properties of classes, etc.: ``package.module.OuterClass.InnerClass.method.__name__``.

        More precisely, the directive first tries to evaluate the name in `ClipppyYAML.scope` (see `Scopes`_). If a `NameError` or `AttributeError` occurs, it tries to import part of the name as a module and evalueate the rest in its scope. It does that at every possible splitting location (a dot), starting from the right, i.e. prefers long imports to long attribute lookups. For example, for ``astropy.units.astrophys.attoparsec``, assuming ``astropy.units`` has not been imported, it will try to import ``astropy.units.astrophys.attoparsec`` first; this will fail, so it will try ``astropy.units.astrophys``, which will have been imported as usual; finally `\!py` will look up the name ``attoparsec`` in the imported module and thus succeed. If nothing works, a `NameError`/`AttributeError`/`ModuleNotFoundError` is raised as appropriate.

        Once the name is resolved, there are two options. If the node is (that is, if there is no value following the NAME), the value of the node is set to the resolved Python object. For example, ::

            key: !py:print

        will parse as ``{'key': <built-in function print>}``. Beware of this style since in YAML a tag *must* be followed by whitespace or end-of-line/transmission, so things like ``{key: !py:print}`` are not valid (just needs a space, though).

        If the node does have a value, the object corresponding to the tag **will be called** with the node contents as arguments, and the node's value will be set to the returned object. Thus, ::

            key: !py:print Making progress!

        will actually print ``Making progress!`` while parsing and return ``{'key': None}`` (since `print` returns `None`). If the node is a scalar, it will be passed as a single argument; if it is a sequence, it will be expanded as ``func(*args)``, and if it is a mapping, as ``func(**kwargs)``, so you can do some wacky things like ::

            !py:str.join [" ", [Hooray, for, Python!]]

        which is the same as ``str.join(' ', ['Hooray', 'for', 'Python!'])``, or outright lose it:

        .. code-block:: yaml
            :name: sorted-example

            !py:sorted
                <: [[[a, 42], [c, 26], [b, 13]]]
                key: !py:operator.itemgetter [0]
                <<: {reverse: True}


        (equivalent to the Python

        .. code-block:: python3

            sorted(*[[['a', 42], ['c', 26], ['b', 13]]],
                   key=operator.itemgetter(0), **{'reverse': True})

        that results in reverse sorting the list by the first entry in each element: ``[['c', 26], ['b', 13], ['a', 42]]``). You can see that the syntax resembles real Python code as close as possible, with the exception of parameter expansions being effected by ``<`` and ``<<`` instead of, respectively, ``*`` and ``**`` (because a ``*`` is reserved for anchors in YAML). The tricks that `\!py` hides up its sleeve are described in full detail in `From Node to Signature`_.

    ``!import``
        A directive that realises Python imports. This tag expects an array node and returns ``None`` as the node's value. Each element node should be a simple string as you would write in Python, and all import styles are supported. The general syntax, therefore, is ::

            whatever name: !import
                - import torch
                - import numpy as np
                - from matplotlib import pyplot as plt

        Loading this config will result in ``{'whatever name': None}``, but as a side effect the respective modules / names will be imported by the standard Python machinery and will be available to *subsequent* `\!py` and `\!eval` directives for name lookup, as well as in `sys.modules`. This directive is primarily useful for ``as``-style imports, abridging qualified names to just the proper ``__name__`` or for making names available in `\!eval`. Other cases are covered by the name resolutionmsemantics of `\!py`.

        .. note:: Additional formats are supported for backwards compatibility. These will not be documented in order to encourage the more sensible standard syntax but can be deduced by perusing the source code of `import_`. I'll give away just that things like ``!import numpy as np`` work as well.

        .. note:: All of the imports from `clipppy.yaml` are already available in YAML and don't need to be imported explicitly for the parsing. This includes `numpy` (as ``np``), `torch`, `io`, `os`, as well as the majority of the |Clipppy| API.

    ``!eval``
        Evaluate the node contents as a Python expression. Basically, this is God mode, although you're still limited to a single expression (not even a statement) since the contents are simply passed on to the built-in `python:eval` function. But a Python God is supposed to be able to do anything in a single expression\ |citation needed|.

    ``!txt``
        Load a text file with numerical data. This is a thin wrapper around `numpy.loadtxt` and as such expects the contents of the node to be valid arguments for it: see `From Node to Signature`_. The particular most frequently used signatures are ::

            !txt data.txt

        and ::

            !txt {fname: data.csv, delimiter: ","}

        The quotation marks are necessary here because a comma is a special character in YAML/JSON.

    ``!npy``
        Load a ``npy`` file. Expects the filename as string and simply passes it on to `numpy.load`::

            !npy data.npy

    ``!npz``
        Load a ``npz`` file. This expects a filename as a string (which will be passed to `numpy.load`) and an optional key (again a string). If the key is given, The particular file from the ``npz`` archive will be returned (see `numpy.savez`). Thus, ::

            !npz [data.npz, somekey]  # or {fname: data.npz, key: somekey}

        is the same as ``numpy.load('data.npz')['somekey']``. Otherwise the opened ``NpzFile`` file will be returned as is::

            !npz data.npz  # same as np.load('archive.npz')

    ``!pt``
        Load a PyTorch archive. As for `\!npz`, one can provide a ``key`` parameter in order to get a specific element from the loaded object. (Note that `torch.load` can save any Python object, so it is not guaranteed that indexing ``torch.load('data.pt')['somekey']`` is sensible.) This function, however, accepts additional (keyword only!) arguments that will be passed on to `torch.load`::

            - !pt data.pt             # torch.load('data.pt')
            - !pt [data.pt, somekey]  # torch.load('data.pt')[somekey]
            - !pt                     # torch.load('data.pt', map_location='cuda', **kwargs)['somekey']
                fname: data.pt
                key: somekey  # optional
                map_location: cuda
                # any other keyword arguments will go into kwargs

    ``!tensor``
        Explicitly construct a `torch.Tensor` via the `torch.tensor` function. The simplest use case is to convert a list of numbers\ [#]_ to a tensor::

            !tensor [[1, 2, 3, 4, 5]]

        Notice the **double brackets**: this is necessary because the node contents first have to be translated to a tuple of arguments, the first of which happens to be an array. Additional (keyword! as per the signature of `torch.tensor`) arguments for the dtype, device, gradient and pinnedness of the tensor are accepted, and furthermore, the ``data`` argument can be an arbitrary construction:

        .. parsed-literal::

            !tensor
                data: `\!npz` [data.npz, somekey]
                dtype: `\!py`:`torch.get_default_dtype` [] [#dtypebrackets]_
                device: cuda

        The above example loads a NumPy array, converts it to the default float type, and puts it on the GPU.

        .. note:: The usual caveats of `torch.tensor` apply. In particular, a copy is **always** made, even if the ``data`` is a `~torch.Tensor` with the requested properties. Furthermore, if an explicit ``device`` argument is not given, any non-`~torch.Tensor` ``data`` will be (copied and) placed on the default ``torch`` device, whereas a `~torch.Tensor` will be (copied and) **kept on the same device**. Use, therefore, :term:`\!tensor:default <!tensor:DTYPE>` to ensure that the result is placed on the default device.

        .. [#] Arguably, it's simpler to convert a single number to a tensor: ``!tensor 42``. This also works but is slightly frowned upon.
        .. [#dtypebrackets] If you're confused about the brackets here, remember that `torch.get_default_dtype` is a function and needs to be called with no arguments.

    ``!tensor:DTYPE``
        In order to simplify the above code, |Clipppy| supports a namespace/prefixed version as a succint way of specifying the desired `Tensor.dtype <torch.torch.dtype>`. This is equivalent to ::

            !tensor
                ...
                dtype: !py:torch.DTYPE

        Acceptable versions, therefore, are ``!tensor:int``, ``!tensor:float``, ``!tensor:double``, ``!tensor:bool``, among others, and the special value, ``!tensor:default``, which stands for the current default dtype **and device** obtained as above.

        .. seealso:: `torch.get_default_dtype`, `torch.set_default_dtype`, `torch.set_default_tensor_type`

    ``!Stochastic:NAME``
        A shortcut for

        .. parsed-literal::

            !py:`clipppy.stochastic.stochastic`
                ...
                name: NAME

        therefore, see the documentation of `~clipppy.stochastic.stochastic`. ``NAME`` (and the colon ``:``) can be omitted and will default to ``None``. Since `~clipppy.stochastic.stochastic` takes at least two arguments, the first one being an object to "wrap" and the second a dictionary of parameter "specifications", the usual YAML pattern is ::

            !Stochastic:NAME
                - !py:MyDeterministicCallable
                    ...  # constructor arguments
                - param_1: ...  # Sampler, etc. or distribution or constant
                  param_2: ...
                  ...

        .. note:: Built into `clipppy.stochastic.stochastic` are two features that make describing stochastic wrappers in YAML (and not only) easier. Firstly, if any of the ``specs.values()`` is an instance of `AbstractSampler` (this includes instances of `Sampler` and company), its name is set to the name of the parameter it is attached to (via `AbstractSampler.set_name`). Secondly, if it is a `Distribution <pyro.distributions.torch_distribution.TorchDistributionMixin>`, a `Sampler` is automatically created from it. This allows for the rather concise

    ``!Param``
    ``!Sampler``
    ``!InfiniteSampler``
    ``!SemiInfiniteSampler``
        Shortcuts for `Param`, `Sampler`, `InfiniteSampler`, `SemiInfiniteSampler`.


As a final shortcut, |Clipppy|'s YAML processor is set up so that by default the top-level node is auto-interpreted as a `Clipppy` object, i.e. it is assigned a tag ``!py:Clipppy``. If this is not desired, use the ``interpret_as_Clipppy`` parameter to `loads`/`load_config` and `ClipppyYAML.load` or explicitly tag the whole document however you like.