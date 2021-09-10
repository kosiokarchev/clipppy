*************************
YAML configuration system
*************************
.. highlight:: yaml

|Clipppy|'s configuration semantics were derived from its venerable predecessor `pyrofit`_, which used YAML files with a more or less predefined structure to define VI models (and guides, etc.). In contrast, |Clipppy|'s YAML "format" is general and can in principle represent the initialisation of any Python objects (and even a bit beyond). The backbone of YAML parsing in |Clipppy| is `ruamel.yaml`_, itself derived from `PyYAML <https://pyyaml.org/>`_, and the very interested reader is referred to their documentations and to the `official YAML specification <https://yaml.org/spec/1.2/spec.html>`_. The less patient among you might be interested in any of a number of YAML tutorials on the web, while if you just want to start off with |Clipppy|, simply read on.

The main entry point functions for YAML parsing in |Clipppy| are `clipppy.loads` (general purpose) and `clipppy.load_config` (for VI). The examples in this guide assume the YAML is loaded via the former, which makes no prior assumptions on the overall structure of the document (while the latter by default interprets it as a `clipppy.Clipppy` object) and works with plain strings (whereas `load_config` requires a path, pathname, or text stream as input).

.. include:: /userguide/yaml/basic.rst

.. include:: /userguide/yaml/signature.rst

.. include:: /userguide/yaml/templating.rst
