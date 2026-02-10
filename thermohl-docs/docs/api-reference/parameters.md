<!---
SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
SPDX-License-Identifier: MPL-2.0
--->

# Parameters and Default Values

Solvers in ThermOHL take a dictionary as an argument, where all keys
are strings and all values are either integers, floats or 1D
`numpy.ndarray` of integers or floats. It is important to note that
all arrays should have the same size.  Missing or `None` values in the
input dictionary are replaced with a default value, available using
`solver.default_values()`.

Below is a table with all physical parameters used in ThermOHL, with
units, default values and in which set of power terms they are used.

[TODO : check with sources + add a paragraph explaining which parameters are not required if we use precomputed_solar_radiation directly]: #

| Parameter | Default Value | Unit       | Used in CIGRE | Used in IEEE | Used in OLLA | Used in RTE | Comment                                                      |
|-----------|---------------|------------|---------------|--------------|--------------|-------------|--------------------------------------------------------------|
| latitude_deg       | 45            | degree     | yes           | yes          | yes          | yes         | latitude                                                     |
| longitude_deg       | 0             | degree     | no            | no           | no           | no          | longitude                                                    |
| altitude       | 0             | m          | yes           | yes          | yes          | yes         | altitude                                                     |
| azimuth       | 0             | degree     | yes           | yes          | yes          | yes         | azimuth                                                      |
| month     | 3             | N/A        | yes           | yes          | yes          | yes         | month number (int in [1, 12])                                |
| day       | 21            | N/A        | yes           | yes          | yes          | yes         | day of the month (int in [1, 31])                            |
| hour      | 12            | N/A        | yes           | yes          | yes          | yes         | hour of the day (float in[0, 24[)                            |
| ambient_temperature_c        | 15            | celsius    | yes           | yes          | yes          | yes         | ambient temperature                                          |
| wind_speed_ms        | 0             | m.s⁻¹      | yes           | yes          | yes          | yes         | wind speed                                                   |
| wind_angle_deg        | 90            | degree     | yes           | yes          | yes          | yes         | wind angle (regarding north)                                 |
| albedo        | 0.8           | N/A        | yes           | no           | no           | no          | albedo                                                       |
| tb        | 0.1           | N/A        | no            | yes          | no           | no          | coefficient for air pollution from 0 (clean) to 1 (polluted) |
| transit   | 100           | A          | yes           | yes          | yes          | yes         | transit intensity                                            |
| m         | 1.5           | kg.m⁻¹     | yes           | yes          | yes          | yes         | mass per unit length (only used in transient mode)           |
| core_diameter_m         | 1.9E-02       | m          | no            | no           | yes          | yes         | core diameter                                                |
| outer_diameter_m         | 3.0E-02       | m          | yes           | yes          | yes          | yes         | external (global) diameter                                   |
| core_area_m2         | 2.84E-04      | m²         | no            | no           | yes          | yes         | core section                                                 |
| A         | 7.07E-04      | m²         | no            | no           | yes          | yes         | external (global) section                                    |
| roughness_ratio         | 4.0E-02       | N/A        | yes           | no           | no           | no          | roughness                                                    |
| l         | 1.0           | W.m⁻¹.K⁻¹  | no            | no           | yes          | yes         | radial thermal conductivity (not used in 1t equation)        |
| c         | 500           | J.kg⁻¹.K⁻¹ | yes           | yes          | yes          | yes         | specific heat capacity (only used in transient mode)         |
| solar_absorptivity     | 0.5           | N/A        | yes           | yes          | yes          | yes         | solar absorption                                             |
| emissivity   | 0.5           | N/A        | yes           | yes          | yes          | yes         | emissivity                                                   |
| dc_resistance_20c_ohm_m     | 2.5E-05       | Ohm.m⁻¹    | yes           | no           | yes          | yes         | electric resistance per unit length (DC) at 20°C             |
| magnetic_coeff        | 1.006         | N/A        | yes           | no           | yes          | yes         | coefficient for magnetic effects                             |
| magnetic_coeff_per_a        | 0.016         | A⁻¹        | no            | no           | yes          | yes         | coefficient for magnetic effects                             |
| temperature_coeff_linear        | 3.8E-03       | K⁻¹        | yes           | no           | yes          | yes         | linear resistance augmentation with temperature              |
| temperature_coeff_quadratic        | 8.0E-07       | K⁻²        | no            | no           | yes          | yes         | quadratic resistance augmentation with temperature           |
| RDCHigh   | 3.05E-05      | Ohm.m⁻¹    | no            | yes          | no           | no          | electric resistance per unit length (DC) at THigh            |
| RDCLow    | 2.66E-05      | Ohm.m⁻¹    | no            | yes          | no           | no          | electric resistance per unit length (DC) at TLow             |
| THigh     | 60            | celsius    | no            | yes          | no           | no          | temperature for RDCHigh measurement                          |
| TLow      | 20            | celsius    | no            | yes          | no           | no          | temperature for RDCHigh measurement                          |
| precomputed_solar_radiation      | NaN           | W.m⁻²      | yes (opt.)    | yes (opt.)   | yes (opt.)   | yes (opt.)  | solar irradiance                                             |

For consistent joule heating outputs between CIGRE and IEEE joule
power terms, you must have

* $ R_{\text{DC,low}}=R_{\text{DC},20} $;
* $ T_{\text{low}}=20 $;
* any $ T_{\text{high}} > T_{\text{low}} $ and
$ R_{\text{DC,high}} - R_{\text{DC,low}} = (T_{\text{high}} - T_{\text{low}}) \cdot k_{\ell} \cdot R_{\text{DC},20} $.

If you use direct solar radiation formula (with `precomputed_solar_radiation` key), you can
ignore the following parameters : `latitude_deg`, `longitude_deg`, `month`, `day`,
`hour`, `albedo` and `tb`.

