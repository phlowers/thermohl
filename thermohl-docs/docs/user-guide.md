<!--
SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
SPDX-License-Identifier: MPL-2.0
-->

# User Guide

## Introduction

The temperature of overhead line conductors is determined by an equilibrium 
between heating and cooling phenomena. Several factors affect this equilibrium.
Here we will have a word about the most common sources of heating and cooling.

All these phenomena correspond to an object `PowerTerm` in the code. If you 
want to use one which is not included or modify an existing one, you can define
your own power term.

### Joule Heating 

Joule heating is the result of the flowing through an imperfect conductor. The
dissipated power is proportional to the conductor's electric resistance and to 
the square of the current ($ P=RI^2 $). The conductor's electric resistance 
usually increases with the conductor temperature. Depending on the conductor 
composition, some magnetic effects can occurs and may be taken into account.

### Solar Heating

Sunlight directly heats the conductor. The absorbed power depends on the 
relative position of the sun regarding the conductor, the solar irradiance, and
the conductor surface properties through an absorption coefficient (darker 
conductors absorb more heat).

### Convection Cooling

Convection (or convective heat transfer) is the transfer of heat from one place 
to another due to the movement of a fluid (the air). In the case of conductor 
temperature estimation, two types of convection are actually considered: 

* Natural convection : air movement due to temperature difference between the 
air close to and the air far from the conductor (hence a density difference and 
the movement).
* Forced convection : when the cooling mechanism is driven by the wind.

Convection cooling is often the most influential cooling mechanism. Usually both
types or convection are computed and the maximum value is taken.

### Radiative Cooling

When heated above ambient temperature, a conductor emits thermal radiation to 
the cooler surroundings. The energy dissipated is given by the 
[Stefan–Boltzmann Law](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law)
($ P=\sigma\varepsilon T^4 $).

### Evaporative Cooling

This phenomemon occurs when the conductor is wet (rain or snow). If the cable is
hot enough, the evaporation of water draws heat away. It is not included in
standard (CIGRE & IEEE) models but may be introduced in further developments. 
Some work about precipitation cooling can be found 
[here](https://www.sciencedirect.com/science/article/abs/pii/S0378779611001337).


## Heat Equations

### General case

The heat conduction equation is a partial differential equation, that describes
the evolution of temperature within time and space. In the case of an overhead
line conductor, we use it in cylindrical coordinates: 
$ 
\rho c_{\text{p}} \frac{\partial T}{\partial t}
=
\lambda
\left[
\frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial T}{\partial r}\right)
+
\frac{1}{r^2}\frac{\partial^2 T}{\partial \theta^2}
+
\frac{\partial^2 T}{\partial z^2}
\right]
$ .

### Simplification for overhead conductors

Overhead line conductors are typically modeled with simplifying assumptions:

* No angular dependency ($ \partial/\partial\theta=0 $) ;
* No axial dependency ($ \partial/\partial z=0 $) .

The previous equation is reduced to a radial, one-dimensional heat equation.
When adding the sources (joule heating), boundary and initial conditions, we
eventually have :

* heat equation : $ \rho c_{\text{p}} \frac{\partial T}{\partial t} = \lambda
  \frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial T}{\partial r}\right) +
  p_{\text{joule}} $ ;
* boundary conditions : $ \frac{\partial T}{\partial r}=0 $ for $ r=0 $ and
  $ -\frac{\partial T}{\partial r}= \frac{q_{\sigma}}{\lambda} $ for $ r=R $ ;
* $ T(r, t=0) = T^0(r) $ a given initial condition.

In this problem:

* $ T $ is the temperature (in Kelvin);
* $ t $ is the time (in seconds);
* $ \rho $ the conductor volumic mass (in kg.m-3);
* $ c_{\text{p}} $ the conductor specific heat capacity (in J.K-1.kg-1);
* $ \lambda $ the conductor heat conductivity (in W.m-1.K-1);
* $ p_{\text{joule}} $ the volumic heat source term (in W.m-3);
* $ q_{\sigma} $ the heat exchange terms (usually the sum of solar heating, 
convective and radiative cooling, in W.m-1).

This equation can be solved with numerical methods (eg FEM), but in order to
apply it to a full electrical network, some simplifications must be carried.

### Heat Equations in ThermOHL

#### Single-temperature model

If we multiply the above equation by $ r $, integrate it before multiplying by 
two over the squared conductor radius, we have the formula of the *average* 
temperature for the conductor. If we assume that the heat terms 
$ p_{\text{joule}} $ and $ q_{\sigma} $ do not depend on $ r $, we have the 
simplified problem 

$ 
\rho c_{\text{p}} \frac{\partial \bar{T}}{\partial t} = p_{\text{joule}} - \frac{1}{\pi R^2} q_{\sigma} 
$ ,

which can be rewritten under the more classic form (we multiply by $ \pi R^2 $, 
then the volumic mass becomes a lineic mass $ m $, the volumic joule term become 
the classic joule term $ q_{\text{joule}} $, and we expanded $ q_{\sigma} = 
-q_{\text{solar}} + q_{\text{convection}} + q_{\text{radiation}} $) :

$ 
m c_{\text{p}} \frac{\partial T}{\partial t} = q_{\text{joule}} + 
q_{\text{solar}} - q_{\text{convection}} - q_{\text{radiation}} 
$ .

In steady mode, we have the even simpler power balance equation (power terms 
may depend on temperature $ T $) : $ q_{\text{joule}} + 
q_{\text{solar}} - q_{\text{convection}} - q_{\text{radiation}} $.

#### Three-temperatures model

In order to have a more accurate resolution than the single-temperature model, 
but with faster solving time than the full resolution, RTE developed a
specific model with three temperatures for the conductor :

* the surface temperature;
* the average temperature;
* the core temperature.
