---
title: "``Clipppy``: A convenience layer for inference and probabilistic programming in Python"
authors:
  - name: Konstantin Karchev
    orcid: 0000-0001-9344-736X
    affiliation: "1, 2"
affiliations:
  - name: |
      SISSA (Scuola Internazionale Superiore di Studi Avanzati)
      via Bonomea 265, I-34136 Trieste, Italy
    index: 1
  - name: |
      Gravitation Astroparticle Physics Amsterdam (GRAPPA),
      Institute for Theoretical Physics Amsterdam and Delta Institute for Theoretical Physics,
      University of Amsterdam, Science Park 904, 1098 XH Amsterdam, The Netherlands
    index: 2
date: 11 February 2026
bibliography: paper.bib
---

# Summary

In the information era, sciences across the board (from string theory to biology) are increasingly concerned with performing *inference*—i.e. extracting insights—from large amounts of complicated empirical observations. To account for the stochasticity inherent in the physical world and for the uncertainty associated with imperfect observations of it, one needs to construct *probabilistic models* for their analysis: either to derive constraints on (make probabilistic "measurements" of) unknown parameters, or to confront multiple competing theories and declare which one(s) is/are in (better) agreement with the data.

The advent of big data has posed a significant computational burden on probabilistic inference. Not only have models grown in sheer size to accommodate ever larger parameter spaces, but have also been constantly increasing in sophistication—physical and statistical—so as to preserve the robustness of results in the face of scientific data of ever improving quality. As a result, an array of modern inference procedures have been developed. Prime examples are Hamiltonian Monte Carlo (HMC) and variational inference (VI), which have recently grown in popularity due to highly performant implementations that make use of two key technologies: massive parallelisation of computations on graphics processing units (GPUs) and automatic differentiation (AD/autograd) of probabilistic models. In order to facilitate their application, a new paradigm has emerged: *probabilistic programming*, in which the model is described as a sequence of stochastic sampling operations—i.e. a forward simulator—rather than in terms of the more traditional explicit probabilities. This has enabled *automation* and *abstraction* of the inference, allowing scientists to seamlessly expand their models in size and complexity. 

# Statement of need

``Clipppy`` facilitates writing probabilistic forward models and performing basic operations with them: generating mock data and parameter inference. ``Clipppy``, further, provides utilities for intervening with the forward simulation and modifying it for the needs of advanced inference algorithms, of which ``Clipppy`` focuses on two: variational inference (VI), for which ``Clipppy`` provides a convenience layer wrapping ``pyro``'s existing functionality, and simulation-based inference (SBI), specifically (truncated marginal) neural ratio estimation ((TM)NRE), of which ``Clipppy`` provides a bespoke implementation.

# Similar software

# Existing applications

``Clipppy`` has been used by @Karchev-vigp,@Karchev-sicret,@Karchev-ptcosmo.

# Acknowledgements

I acknowledge the guidance and supervision of Christoph Weniger, the original author of ``pyrofit``, and would like to also thank Adam Coogan, Noemi Anau Montel, Elias Dubbeldam for their comments and suggestions in developing and improving ``Clipppy``.

# References
