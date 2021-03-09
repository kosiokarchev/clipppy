Welcome to CLIPPPY's documentation!
===================================

CLIPPPY is a Command Line Interface to Probabilistic Programming in PYthon.
If you can think of a better name, let me know!

Its main purpose is to assuage some of the pains I encountered while using
`pyrofit <https://github.com/cweniger/pyrofit-core>`_ while preserving its
good sides. In the end, I opted to rewrite most of the machinery instead of
fixing it up in patches.

Provided with the hope it will be useful but devoid of any guarantees
(except those that can be safely inferred by following strict development logic).


Overview
--------

The package has four main components.
A :mod:`~clipppy.stochastic` module provides
utilities to wrap generative models (or any code, actually) in wrappers that
take care of their stochasticity (i.e. generate new sets of parameters from
predefined (but possibly varying in the course of execution) distributions
each time the code is invoked).
A :mod:`~clipppy.guide` module defines guides classes
that automate much of the manual work (code there "takes" heavy inspiration
from `pyro.contrib.easyguide` and the original ``pyrofit`` guides).
Strictly, one is not forced to use those and can instead provide any callable
as a guide, but a lot of the functionality centres around the
:class:`~clipppy.guide.Guide` interface, so using it is highly recommended. It
should be straightforwardly extendable.
The :class:`~clipppy.Clipppy.Clipppy`
class provides the main routines for performing inference: fitting, mock data
generation, and more on the way!
And finally, the eponimous :mod:`~clipppy.cli` module handles the loading of
YAML files in a sensible? extensible? way so that configurations can be easily
reused, shared, and archived, while providing (almost?) the full flexibility
of a custom script model definition.


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   userguide
   devguide


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
