Install
=======

Using pip
---------
To install the package using pip, execute the following command:

.. code-block::

    python -m pip install git+https://gitlab.eurobios.com/rte/thermohl

Using conda
-----------
(not available yet)

The package is available on conda-forge. To install, execute the following command:

.. code-block::

    python -m conda install thermohl -c conda-forge

Simple usage
===========

More examples are available in the ``examples`` directory.

Solvers in thermOHL take a dictionary as an argument, where all keys are strings and all values are either integers,
floats or 1D `numpy.ndarray` of integers or floats. It is important to note that all arrays should have the same size.
Missing or `None` values in the input dictionary are replaced with a default value, available using
`solver.default_values()`, which are read from `thermohl/default_values.yaml`.

Example 1:
-----------

This example uses the IEEE model with default values to compute the surface temperature (°C) of a conductor in steady
regime along with the corresponding power terms (W.m\ :sup:`-1`) in the Energy Balance Principle.

.. code-block:: python

    from thermohl import solver

    slvr = solver.ieee({})
    temp = slvr.steady_temperature()

Results from the solver are returned in a `pandas.DataFrame`:

.. code-block::

    >>> print(temp)
         T_surf   P_joule  P_solar  P_convection  P_radiation  P_precipitation
    0  27.22858  0.273048  9.64051        6.5819     3.331658              0.0

Example 2:
-----------

This example uses the IEEE model to compute the maximum current intensity (A) that can be used in a conductor without
exceeding a specified maximal temperature (°C), along with the corresponding power terms (W.m\ :sup:`-1`) in the Energy
Balance Principle. Three ambient temperatures are specified as inputs, meaning that three corresponding maximal
intensities will be returned by the solver.

.. code-block:: python

    import numpy as np
    from thermohl import solver

    slvr = solver.ieee(dict(Ta=np.array([0., 15., 30.])))
    Tmax = 80.
    imax = slvr.steady_intensity(Tmax)

.. code-block::

    >>> print(imax)
             I_max
    0  1591.395583
    1  1390.793694
    2  1164.086618
