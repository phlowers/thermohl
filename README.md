# ThermOHL

_**Thermohl**_ is a python package allowing the computation of the temperature in overhead conductors for given
transit intensity and environment parameters, or to compute the maximum possible transit intensity given a maximum
temperature and environment parameters. A collection of models based on the Energy Balance Principle are available.

## Main features

Three different solvers are available in the package:

- computation of the steady-state temperature;
- computation of the transient temperature;
- computation of the maximum possible steady-state transit intensity.

All solvers are based on the Energy Balance Principle, where the various power terms are calculated from one of four
available models: a CIGRE and an IEEE model as well as two RTE-specific models, OLLA and CNER.

Both steady-state solvers can be used for probabilistic simulations with random input parameters using the Monte Carlo
method. Uncertainty quantification can be performed through statistics on the random simulation outputs, as well as
through a sensitivity analysis using Sobol indices which allows ranking random input parameters according to their
contribution to the total variance of the output.

## Installation

### Using pip

To install the package using pip, execute the following command:

```shell script
python -m pip install slenderpy@git+https://github.com/rte-france/pallas-thermohl
```

### Using conda

(not available yet)

## Building the documentation

First, make sure you have sphinx and the Readthedocs theme installed.

If you use pip, open a terminal and enter the following commands:

```shell script
pip install sphinx
pip install sphinx_rtd_theme
```

If you use conda, open an Anaconda Powershell Prompt and enter the following commands:

```shell script
conda install sphinx
conda install sphinx_rtd_theme
```

Then, in the same terminal or anaconda prompt, build the doc with:

```shell script
cd doc
make html
```

The documentation can then be accessed from `doc/_build/html/index.html`.

## Simple usage

Solvers in thermOHL take a dictionary as an argument, where all keys are strings and all values are either integers,
floats or 1D `numpy.ndarray` of integers or floats. It is important to note that all arrays should have the same size.
Missing or `None` values in the input dictionary are replaced with a default value, available using
`solver.default_values()`, which are read from `thermohl/default_values.yaml`.

### Example 1:

This example uses the IEEE model with default values to compute the surface temperature (°C) of a conductor
in steady regime along with the corresponding power terms (W.m<sup>-1</sup>) in the Energy Balance Principle.

```python
from thermohl import solver

slvr = solver.ieee({})
temp = slvr.steady_temperature() 
```

Results from the solver are returned in a `pandas.DataFrame`:

``` python
>>> print(temp)
     T_surf   P_joule  P_solar  P_convection  P_radiation  P_precipitation
0  27.22858  0.273048  9.64051        6.5819     3.331658              0.0
```

### Example 2:

This example uses the IEEE model to compute the maximum current intensity (A) that can be used in a conductor without
exceeding a specified maximal temperature (°C), along with the corresponding power terms (W.m<sup>-1</sup>)
in the Energy Balance Principle. Three ambient temperatures are specified as inputs, meaning that three corresponding
maximal intensities will be returned by the solver.

```python
import numpy as np
from thermohl import solver

slvr = solver.ieee(dict(Ta=np.array([0., 15., 30.])))
Tmax = 80.
imax = slvr.steady_intensity(Tmax)
```

```
>>> print(imax)
         I_max    P_joule  P_solar  P_convection  P_radiation  P_precipitation
0  1606.946066  83.794845  9.64051     66.750785    26.684570              0.0
1  1408.560563  64.382191  9.64051     50.884473    23.138228              0.0
2  1185.256686  45.586844  9.64051     36.234737    18.992617              0.0
```
