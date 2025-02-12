---
title: "[Clipppy]{.sc}: A convenience layer for inference and probabilistic programming in `Python`"
authors:
  - name: Konstantin Karchev
    orcid: 0000-0001-9344-736X
    affiliation: "1, 2"
affiliations:
  - index: 1
    name: TSDS, Scuola Internazionale Superiore di Studi Avanzati (SISSA), Trieste, Italy
    ror: 004fze387
  - index: 2
    name: GRAPPA, University of Amsterdam (UvA), Amsterdam, The Netherlands
    ror: 04dkp9463
date: 11 February 2026
bibliography: paper.bib
---

# Summary

[Clipppy]{.sc} is a collection of `Python` utilities (based on `PyTorch` [@pytorch] and `Pyro` [@pyro]) that facilitate writing high-performance automatically differentiable probabilistic forward models and performing inference with them. It defines a YAML-based language for describing general stochastic procedures and provides tools for inspecting and manipulating their execution as well as fitting them to data through variational inference (wrapping and extending `Pyro`'s functionalities in this respect) and for interoperating with other industry-standard tools. Furthermore, [Clipppy]{.sc} includes a highly customisable framework for simulation-based parameter inference and model comparison, part of which are routines for storing and handling large suites of simulation outputs, defining neural network-based components, tracking and visualising their training progress, and plotting, validating, and calibrating the results with bespoke algorithms. By bringing flexible model building and principled probabilistic inference under a common umbrella, [Clipppy]{.sc} aims to streamline data analysis applications across the natural sciences while also fostering development of cutting-edge computational techniques from the field of probabilistic modelling and inference.

# Statement of need

Probabilistic inference is used in the natural sciences (from particle physics to cosmology to biology) to constrain unknown parameters or confront competing theories (models) while accounting for the stochasticity inherent in the physical world and for the uncertainty associated with imperfect observations of it. However, with the advent of big—and increasingly precise—data, this task has become computationally burdensome due to the sheer size and sophistication—both physical and statistical—of modern scientific models.
 
As a result, an array of modern inference techniques have been developed. Prime examples are hybrid/Hamiltonian Monte Carlo [HMC; @HMC; @Neal_2011] and variational inference [VI; see e.g. @Blei_2017; @Zhang_2019], whose popularity has recently surged due to highly performant implementations that make use of two key technologies: massive parallelisation of the computations on graphics processing units (GPUs) and automatic differentiation (AD/autograd). In order to facilitate their application, a new software paradigm has emerged: *probabilistic programming* (PP), in which the model is described as a sequence of stochastic sampling operations—i.e. a forward simulator. By *automating* the necessary probability calculations, PP *abstracts* the inference procedure and allows scientists to seamlessly expand and improve their forward models to more faithfully represent their studied phenomena.

Meanwhile, borne on the wave of innovations in deep machine learning (ML), a new framework for purely simulation-based inference [SBI; see @Cranmer_2020; @Lueckmann_2021 for reviews] has recently emerged. Rather than explicitly evaluating probabilities, modern SBI methods aim to *learn* them from realistic mock data by training neural networks. This allows for greater modelling flexibility (e.g. to represent sophisticated parameter hierarchies, instrumental effects, and sample-selection procedures), accelerates inference in high-dimensional settings, and enables fast validation and calibration of the results in both Bayesian and frequentist contexts [see e.g. @Hermans_2022; @Karchev-sicret; @Jeffrey_2023; @Karchev-simsims].

# Feature highlights and existing applications

[Clipppy]{.sc}'s functionalities fall in two categories: model authoring and inference. For the former, [Clipppy]{.sc} extends `Pyro`/`PyTorch` with classes for flexible numerically defined distributions (histograms and tesselations), the ability to "truncate"[^truncation] univariate analytic distribution, and utilities for sampling independent and identically distributed variables in parallel[^plate], including batches of unequal sizes. Besides procedurally in `Python` code, [Clipppy]{.sc} allows stochastic programs to be defined in a human-readable and extensively documented YAML format (with optional Ninja templating), which greatly simplifies the creation and utilisation of probabilistic models.

[^truncation]: Restricting the support of priors has emerged as the prevalent form of sequential simulation-based inference [@Miller_2021;@Deistler_2022], aiming to boost data efficiency and better utilise the NN's learning capacity.

[^plate]: This is complementary to (and in the opinion of this author, more sensible than) `Pyro`'s `plate`-ing system.

On the other hand, [Clipppy]{.sc} builds on `Pyro`'s engine for variational inference by enabling the automatic construction of variational approximations ("guides") and controlling fit hyperparameters like the learning rate and number of steps. Through `Pyro`'s tracing mechanism, [Clipppy]{.sc} can also automatically extract the necessary model probabilities and interoperate with external fitting codes: e.g. `emcee` [@emcee] for affine-invariant Markov Chain Monte Carlo and `dynesty` [@dynesty] for nested sampling.

Moreover, [Clipppy]{.sc} includes a suite of standalone tools for simulation-based inference: neural network components and losses for posterior estimation [NPE, @Papamakarios_2016] (including the model-gradient assisted variant [@Zeghal_2022]), likelihood-to-evidence ratio estimation [NRE, @Hermans_2020] (marginal and autoregressive [@AnauMontel_2023]), and model comparison through neural classification [@Karchev-simsims], as well as routines for Bayesian validation and frequentist calibration. The encessary mock data can either be generated on the fly by a "Clipppied" stochastic model or loaded from disk[^storage], while the training loop is embedded in the `PyTorch Lightning` [@ptlightning] ecosystem and can thus benefit from its rich visualisation and hyper-optimisation tools and integrations.

[^storage]: [Clipppy]{.sc} can use memory-efficient storage based on the `zarr` and `netcdf` formats.

[Clipppy]{.sc} has been used by its author and collaborators in an array of publications in the fields of gravitational lensing [@Karchev-vigp], supernova cosmology [@Karchev-ptcosmo;@Karchev-sicret;@Karchev-sidereal;@Karchev-simsims;@Karchev-starnre], and exoplanet research [@Benito_2024;@Lueber_2025].

# Related software

The probabilistic-programming paradigm has been realised in a growing number of dedicated languages, the most prominent of which being [Bugs]{.sc} [@BUGS], `Stan` [@stan;@stan-autograd], `Turing.jl` [@Turing.jl].[^pp-list] PP *frameworks*, embedded within general-purpose languages, include `PyMC` [@pymc], `Pyro` [@pyro], `NumPyro` [@numpyro], and `TensorFlow Probability` (which use, respectively, `PyTensor` [@pytensor], `PyTorch` [@pytorch], `JAX` [@jax], and `TensorFlow` [@tensorflow] for GPU computations and automatic differentiation). On the other hand, in the field of SBI, distinct algorithms previously implemented in self-contained software: e.g. [`PyDelfi`](https://github.com/justinalsing/pydelfi) and `Swyft` [@swyft], have now been centralised in the `sbi` package [@sbi].

[^pp-list]: A more comprehensive list is compiled on [Wikipedia](https://en.wikipedia.org/wiki/Probabilistic_programming#List_of_probabilistic_programming_languages).

[Clipppy]{.sc} is based on the `PyTorch`+`Pyro` stack due to its maturity and large collection of ML- and NN-related utilities, as well as its high performance, versatility, and low development and deployment burden[^jit]. Distinctly from the alternatives above, [Clipppy]{.sc} aims to be a one-stop solution for the implementation of *and* inference (likelihood- *and* simulation-based) with probabilistic models. It emphasises reusability—of software components across implementations and model definition across inference methods—, flexibility, and extensibility so as to accommodate new analysis techniques and ever-increasing modelling sophistication.

[^jit]: in comparison with frameworks with just-in-time compilation (e.g. `JAX`), which tend to be more difficult to install, in this author's experience

# Acknowledgements

I acknowledge the guidance and supervision of Christoph Weniger and Roberto Trotta and would also like to thank my collaborators and early [Clipppy]{.sc} adopters Adam Coogan, Noemi Anau Montel, Elias Dubbeldam, María Benito Castaño, and Anna Lueber for their comments and suggestions in developing and improving the software.

# References
