Welcome to |Clipppy|'s documentation!
=====================================

|Clipppy| is a Convenience Layer for Inference and Probabilistic Programming in PYthon. If you can think of a better name, let me know!

Inspired by the venerable `pyrofit`_, |Clipppy| does the boring things automatically while staying out of the way of advanced users.

Provided with the hope it will be useful but devoid of any guarantees (except those that can be safely inferred by following strict development logic).


Overview
--------

The package has four main components. A :mod:`~clipppy.stochastic` module provides utilities to wrap generative models (or any code, actually) in wrappers that take care of their stochasticity (i.e. generate new sets of parameters from predefined (but possibly varying in the course of execution) distributions each time the code is invoked). A :mod:`~clipppy.guide` module defines guides classes that automate much of the manual work (code there "takes" heavy inspiration from `pyro.contrib.easyguide` and the original `pyrofit`_ guides). Strictly, one is not forced to use those and can instead provide any callable as a guide, but a lot of the functionality centres around the :class:`~clipppy.guide.Guide` interface, so using it is highly recommended. It should be straightforwardly extendable. The :class:`~clipppy.Clipppy.Clipppy` class provides the main routines for performing inference: fitting, mock data generation, and more on the way!
:raw-html:`<strike>`
And finally, the eponimous :mod:`~clipppy.cli` module handles the loading of YAML files in a sensible? extensible? way so that configurations can be easily reused, shared, and archived, while providing (almost?) the full flexibility of a custom script model definition.
:raw-html:`</strike>`


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
