<!--
SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
SPDX-License-Identifier: MPL-2.0
-->

# ThermOHL

[![MPL-2.0 License](https://img.shields.io/badge/license-MPL_2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=phlowers_thermohl&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=phlowers_thermohl)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=phlowers_thermohl&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=phlowers_thermohl)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=phlowers_thermohl&metric=coverage)](https://sonarcloud.io/summary/new_code?id=phlowers_thermohl)

# ThermOHL

Temperature estimation of overhead line conductors is an important topic for 
TSOs for technical, economic, and safety-related reasons (DLR/ampacity, sag 
management ...). It depends on several factors, mainly transit, weather and the
conductor properties. ThermOHL is a python package to compute temperature and/or 
ampacity in overhead line conductors.

## Features

The temperature of a conductor is estimated by solving a heat equation
which describes how temperature evolves over time, taking into account
different power terms that either heat or cold the conductor (see next picture 
from CIGRE[1]).

![image](thermohl-docs/docs/assets/images/cigre_balance.png "Overhead conductor heating and cooling. From [CIGRE].")

Two heat equations (a more complete, third one is under development)
are available:

* one with a single temperature for the cable;
* another with three temperatures (core, average and surface
  temperature) for more precise computations.

Each of these equations can be used with a set of pre-coded power
terms from the literature :

* one using CIGRE recommendations [1];  
* one using the IEEE standard [2];  
* two others from RTE departments.

Solvers derivated from heat equations can compute steady-state
temperature or ampacity, and transient temperature. The set of
[parameter](thermohl-docs/docs/api-reference/parameters.md) required depends on 
the power terms used, and default values are provided.

## References

* [1] Stephen et al., **Thermal behaviour of overhead conductors**. 
  *CIGRE, Study committee 22, working group 12*, 2002.
  https://e-cigre.org/publications/detail/207-thermal-behaviour-of-overhead-conductors.html.
* [2] IEEE, **Standard for Calculating the Current-Temperature Relationship of Bare Overhead Conductors**.
  *IEEE Std 738–2012 (Revision of IEEE Std 738–2006, Incorporates IEEE Std 738–2012 Cor 1–2013)*, 2013.
  https://doi.org/10.1109/IEEESTD.2013.6692858.


## Installation

### Using pip

To install the package using pip, execute the following command:

```shell
    python -m pip install thermohl@git+https://github.com/phlowers/thermohl
```

## Development

Install the development dependencies and program scripts via

```shell
  pip install -e .[dev]
```

Build a new wheel via

```shell
  pip install build
  python -m build --wheel
```

This build a wheel in newly-created dist/ directory

## Building the documentation with mkdocs

First, make sure you have mkdocs and the Readthedocs theme installed.

If you use pip, open a terminal and enter the following commands:

```shell 
  pip install -e .[docs]
```

Then, in the same terminal, build the doc with:

* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

The documentation can then be accessed locally from http://127.0.0.1:8000.

## Simple usage

Solvers in thermOHL take a dictionary as an argument, where all keys are strings and all values are either integers,
floats or 1D `numpy.ndarray` of integers or floats. It is important to note that all arrays should have the same size.
Missing or `None` values in the input dictionary are replaced with a default value, available using
`solver.default_values()`, which are read from `thermohl/default_values.yaml`.

### Example 1

This example uses the single-temperature heat equation (`1t`) with IEEE power terms and default values to compute the
surface temperature (°C) of a conductor in steady-state regime along with the corresponding power terms (W.m<sup>-1</sup>).

```python
from thermohl import solver

slvr = solver.ieee(dic=None, heateq='1t')
temp = slvr.steady_temperature() 
```

Results from the solver are returned in a `pandas.DataFrame`:

``` python
>>> print(temp)
           t   P_joule  P_solar  P_convection  P_radiation  P_precipitation
0  27.236417  0.273056  9.64051      6.587129     3.326436              0.0
```

### Example 2

This example uses the same heat equation and power terms, but to compute the line ampacity (A), ie the maximum power 
intensity that can be used in a conductor without exceeding a specified maximal temperature (°C), along with the 
corresponding power terms (W.m<sup>-1</sup>). We can see that, for three different ambient temperature, we have three
distinct ampacities (and the lower the ambient temperature, the higher the ampacity).

```python
import numpy as np
from thermohl import solver

slvr = solver.ieee(dict(Ta=np.array([0., 15., 30.])), heateq='1t')
Tmax = 80.
imax = slvr.steady_intensity(Tmax)
```

```
>>> print(imax)
             I    P_joule  P_solar  P_convection  P_radiation  P_precipitation
0  1606.398362  83.737734  9.64051     66.750785    26.627459              0.0
1  1408.025761  64.333311  9.64051     50.884473    23.089348              0.0
2  1184.741847  45.547250  9.64051     36.234737    18.953023              0.0
```
