<!--
SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
SPDX-License-Identifier: MPL-2.0
-->

## Users

---

### Environment
ThermOHL is using pip for project and dependencies management.
You need a compatible version of python (3.8 or higher). You may have to install it manually (e.g. with pyenv). Then you may create a virtualenv and activate it.

### Set up thermohl

To install the package using pip, execute the following command:

```shell
    pip install thermohl
```

Use it ! You can report to the user guide section.
```shell
    import thermohl
    print(thermohl.__version__)
```

## Developers

---

Install the development dependencies and program scripts via

```shell
  uv pip install -e .
  uv sync --group dev
```

Build a new wheel via

```shell
  uv build --wheel
```

This build a wheel in newly-created dist/ directory

## Building the documentation with mkdocs

First, make sure you have mkdocs and the Readthedocs theme installed.

If you use uv, open a terminal and enter the following commands:

```shell 
  uv sync --group docs
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
corresponding power terms (W.m<sup>-1</sup>). We can see that, for three different ambient temperatures, we have three
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


