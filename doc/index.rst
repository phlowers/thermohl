.. TermOHL documentation master file, created by
   sphinx-quickstart on Wed Mar 10 16:51:16 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TermoHL's documentation!
===================================

TermOHL is a python module to compute temperature (given environment
parameters) or maximum transit intensity (given a maximum temperature
and environment parameters) in overhead line conductors.

Three temperature models are currently available: one using CIGRE
recommendations, one using an IEEE standard an two other from RTE
(OLLA and CNER). Steady-state versions cover both temperature and
transit solver; concerning transient version, only the temperature is
implemented. Probabilistic simulations with random inputs are only
possible with the steady-state solvers.

About performance
-----------------

The solver is very fast: when using large arrays, an element is
computed in less than 3 microseconds on a single core of an Intel
i7-8700 (a good desktop computer). This is equivalent to solve a set
of 500 spans with hourly time steps for a year in less than 15
seconds!


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   getting-started

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Parameters and default values

   param-defval

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Package Reference

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
