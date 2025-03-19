<!--
SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
SPDX-License-Identifier: MPL-2.0
-->

<img src="_static/logos/thermohl_logo.png" width="200" height="200" alt="Phlowers logo" style="float: right; display: block; margin: 0 auto"/>

# ThermOHL

ThermOHL is a python module to compute temperature (given environment
parameters) or maximum transit intensity (given a maximum temperature
and environment parameters) in overhead line conductors.

## Features

Three temperature models are currently available: 

* one using CIGRE recommendations, 
* one using an IEEE standard,
* two other from RTE (OLLA and CNER). 

Steady-state versions cover both temperature and transit solver; concerning transient version, only the temperature is
implemented. Probabilistic simulations with random inputs are only possible with the steady-state solvers.

## Examples
Todo
