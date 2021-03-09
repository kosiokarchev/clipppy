####################
Clipppy Users' Guide
####################

Clipppy's features fall into two broad categories: a configuration system and runtime utilities. Whereas the latter are aimed mainly at probabilistic programming (PP) and variational inference (VI), the former is meant to be as general as possible in its core while having specific "shortcuts" useful in PP and VI.


*************************
YAML configuration system
*************************
.. highlight:: yaml

Clipppy's configuration semantics were derived from its venerable predecessor ``pyrofit.core``, which used YAML files with a more or less predefined structure to define VI models (and guides, etc.). In contrast, Clipppy's YAML "format" is general and can in principle represent the initialisation of any Python objects. The backbone of YAML parsing in Clipppy is `ruamel.yaml <https://yaml.readthedocs.io/en/latest/>`_, and the main entry point function is `clipppy.load_config`. The very interested reader is referred to the `official YAML specification <https://yaml.org/spec/1.2/spec.html>`_, while less patient ones might be interested in any of a number of YAML tutorials on the web. If you just want to start off with Clipppy, though, read on.

.. note::
    In the following, we'll assume that a given example YAML file has been loaded into Python as::

        config = clipppy.load_config('config.yaml')

Basic YAML
==========

YAML is [#aint]_ a markup language that extends JSON. Hence, any valid JSON, like ``{"answer": 42, "foo": [3.14, "euler's number", {"question": null, "whatever": {"maybe": true, "but actually": false}}]}`` is valid YAML. This basic variant (which is almost directly usable as a Python literal) allows the description of arbitrary primitive objects: numbers, strings, arrays (Python lists), and dictionaries. YAML also allows a modified syntax, where one is permitted to

- ditch the quotation marks since everything that doesn't look like a number (and isn't a literal like ``true``, ``false``, ``null``) is interpreted as a string;
- ditch the brackets as long as one uses indentation: everything more indented with respect to the parent by the same amount is on the same nesting level;
- use a bullet-point-like style for lists (using ``-``) instead of square brackets.

Thus, in YAML the above example may be rewritten as::

    answer: 42
    foo:
      - 3.14
      - euler's number
      - question: null
        whatever: {"maybe": true, but actually: false}

which is, arguably, way more pleasant to look at. [#fretnot]_ Note that one can still use (even partially well-formatted) JSON for any node.

.. note:: YAML also permits using lists (will be converted to tuples) as keys in a dictionary: ::

        [a string, 26]: value

    However, the usual hashability rules of Python apply, so dictionaries are not allowed inside keys. Also, since non-string keys are not used in function definitions, this feature is discouraged in Clipppy, and no guarantees are made that it will indeed be allowed forever.

.. rubric:: Footnotes
.. [#aint] according to its name, though, it ain't
.. [#fretnot] Fret not! It will soon become much messier!

Advanced YAML [#intermediate]_
==============================

On top of syntactic sugar, YAML comes with some useful additional features. One of them is the ability to name and subsequently reference nodes. The syntax is inspired by C's pointers: ::

    a: *name {key1: value1, key2: value2}
    b: &name

Using ``*name`` defines the "variable", and ``&name`` "dereferences" it [#pointers]_. The pointer language is accurate here since in the parsed object, the two nodes will be converted to *references to the same object*, so ``parsed['a'] is parsed['b']`` will evaluate to ``True`` in Python. Since this is a standard feature of ``ruamel.yaml``, Clipppy's machinery is bypassed when dereferencing, which might be surprising to someone who uses YAML references as a way to avoid duplicating code and doesn't really mean to have the same object.

.. rubric:: Footnotes
.. [#intermediate] well, intermediate, really
.. [#pointers] The correct terms in YAML-speak are "anchor" and "alias".

Tags
----

Secondly, YAML introduces the idea of *tags* (and Clipppy takes it maybe a bit too close to the heart). They are identified by a prefixed ``!``, as in ``!tag``, and, as in any markup language, define the type of the node/element. They are metadata and denote a step of postprocessing the primitive values found in the node contents.

.. note:: The specification goes on and on about tags and prefixes and namespaces and URIs and local and global and what-not tags... Truth is, every node must have a tag (be it an empty one), but end users usually don't bother them since most tags are assigned automatically by the parser based on whether a node is a number, a string, a literal, an array, or a dictionary. Clipppy adds specific tags for specific object types and uses other tags to enable more complex behaviour presented below. Tags are also assigned (experimentally) based on type annotations as described in `The Power of Type Annotations`_.

.. glossary::

    ``!py``
        ProbablY the most common Clipppy tag allows to access arbitrary objects defined in the Python scope from which the YAML is loaded.

        .. warning:: Write this

    ``!import``
        A directive that realises Python imports. This tag expects an array node and returns ``None`` as the node's value. Each element node should be a simple string as you would write in Python, and all import styles are supported. The general syntax, therefore, is::

            whatever name: !import
                - import torch
                - import numpy as np
                - from matplotlib import pyplot as plt

        Loading this config will result in ``{'whatever name': None}``, but as a side effect the respective modules / names will be imported by the standard Python machinery and will be available in the scope from which the YAML is loaded, as well as in `sys.modules`. They will also be available to *subsequent* :term:`\!py` directives for name lookup.

        .. note:: Additional formats are supported for backwards compatibility. These will not be documented in order to encourage the more sensible standard syntax, but can be deduced by perusing the source code of `clipppy.yaml._import`.

        .. note:: All of the imports from `clipppy.yaml` are already available in the YAML and don't need to be imported explicitly for the parsing. This includes ``numpy`` (as ``np``), ``torch``, ``io``, ``os``, as well as the majority of the Clipppy API.

    ``!eval``
        evaluate the node contents as a Python expression. Basically, this is God mode, although you're still limited to a single expression (not even a statement) since the contents are simply passed on to the built-in `eval` function. But a Python God is supposed to be able to do anything in a single expression :superscript:`[citation needed]`.

    ``!txt``
        Load a text file with numerical data. This is a thin wrapper around `numpy.loadtxt` and as such expects the contents of the node to be valid arguments for it: see `From Signature to Node`_. The particular most frequently used signatures are::

            !txt data.txt

        and::

            !txt {fname: data.csv, delimiter: ","}

        The quotation marks are necessary here because a comma is a special character in YAML/JSON.

    ``!npy``
        Load a ``npy`` file. Expects the filename as string and simply passes it on to `numpy.load`: ::

            !npy data.npy

    ``!npz``
        Load a ``npz`` file. This expects a filename as a string (which will be passed to `numpy.load`) and an optional key (again a string). If the key is given, The particular file from the ``npz`` archive will be returned (see `numpy.savez`). Thus,::

            !npz [data.npz, somekey]  # or {fname: data.npz, key: somekey}

        is the same as ``numpy.load('data.npz')['somekey']``. Otherwise the opened ``NpzFile`` file will be returned as is: ::

            !npz data.npz  # same as np.load('archive.npz')

    ``!pt``
        Load a PyTorch archive. As for :term:`\!npz`, one can provide a ``key`` parameter in order to get a specific element from the loaded object. (Note that `torch.load` can save any Python object, so it is not guaranteed that indexing ``torch.load('data.pt')['somekey']`` is sensible.) This function, however, accepts additional (keyword only!) arguments that will be passed on to `torch.load`: ::

            - !pt data.pt             # torch.load('data.pt')
            - !pt [data.pt, somekey]  # torch.load('data.pt')[somekey]
            - !pt                     # torch.load('data.pt', map_location='cuda', **kwargs)['somekey']
                fname: data.pt
                key: somekey  # optional
                map_location: cuda
                # any other keyword arguments will go into kwargs

    ``!tensor``
        Explicitly construct a `torch.Tensor` via the `torch.tensor` function. The simplest use case is to convert a list of numbers to a tensor: ::

            !tensor [[1, 2, 3, 4, 5]]

        Notice the **double brackets**: this is necessary because the node contents first have to be translated to a tuple of arguments, the first of which happens to be an array. Additional (keyword! as per the signature of `torch.tensor`) arguments for the dtype, device, gradient and pinnedness of the tensor are accepted, and furthermore, the ``data`` argument can be an arbitrary construction:

        .. parsed-literal::

            !tensor
                data: :term:`\!npz` [data.npz, somekey]
                dtype: :term:`\!py`:`torch.get_default_dtype` [] [#dtypebrackets]_
                device: cuda

        The above example loads a NumPy array, converts it to the default float type, and puts it on the GPU.

        .. note:: The usual caveats of `torch.tensor` apply. In particular, a copy is **always** made, even if the ``data`` is a `Tensor <torch.Tensor>` with the requested properties. Furthermore, if an explicit ``device`` argument is not given, any non-`Tensor <torch.Tensor>` ``data`` will be (copied and) placed on the default ``torch`` device, whereas a `Tensor <torch.Tensor>` will be (copied and) **kept on the same device**. Use, therefore, :term:`\!tensor:default <!tensor:DTYPE>` to ensure that the result is placed on the default device.

        .. [#dtypebrackets] If you're confused about the brackets here, remember that `torch.get_default_dtype` is a function and needs to be called with no arguments.

    ``!tensor:DTYPE``
        Remember namespaces, etc. for tags? In order to simplify the above code, Clipppy supports this syntax as a succint way of specifying the desired `Tensor.dtype <torch.Tensor.dtype>`. This is equivalent to::

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

        therefore, see the documentation of `stochastic <clipppy.stochastic.stochastic>`. ``NAME`` (and the colon ``:``) can be omitted and will default to ``None``. Since `stochastic <clipppy.stochastic.stochastic>` takes at least two arguments, the first one being an object to "wrap" and the second a dictionary of parameter "specifications", the usual YAML pattern is::

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
        Sfortcuts for `Param`, `Sampler`, `InfiniteSampler`, `SemiInfiniteSampler`.


As a final shortcut, Clipppy's YAML processor is set up so that by default the top-level node is auto-interpreted as a `Clipppy <clipppy.Clipppy.Clipppy>` object, i.e. it is assigned a tag ``!py:clipppy.Clipppy``, and its element named ``guide`` is tagged with ``!py:``\ `clipppy.guide.guide.Guide`. If this is not desired, use the ``interpret_as_Clipppy`` parameter to `load_config` and `ClipppyYAML` or directly declare the YAML a dictionary or list using ``!py:dict`` or ``!py:list``, respectively, at the top of the file.

The Power of Type Annotations
=============================

From Signature to Node
----------------------
