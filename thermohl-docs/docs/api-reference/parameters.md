<!---
SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
SPDX-License-Identifier: MPL-2.0
--->

# Parameters and Default Values

| Parameter | Default Value | Unit    | Used in CIGRE | Used in IEEE | Used in OLLA | Used in CNER | Comment                                                      |
|-----------|---------------|---------|---------------|--------------|--------------|--------------|--------------------------------------------------------------|
| lat       | 45            | degree  | yes           | yes          | yes          | yes          | latitude                                                     |
| lon       | 0             | degree  | no            | no           | no           | no           | longitude                                                    |
| alt       | 0             | m       | yes           | yes          | yes          | yes          | altitude                                                     |
| azm       | 0             | degree  | yes           | yes          | yes          | yes          | azimuth                                                      |
| month     | 3             | N/A     | yes           | yes          | yes          | yes          | month number ([1, 12])                                       |
| day       | 21            | N/A     | yes           | yes          | yes          | yes          | day of the month ([1, 31])                                   |
| hour      | 12            | N/A     | yes           | yes          | yes          | yes          | hour of the day ([0, 23])                                    |
| Ta        | 15            | celsius | yes           | yes          | yes          | yes          | ambient temperature                                          |
| ws        | 0             | m/s     | yes           | yes          | yes          | yes          | wind speed                                                   |
| wa        | 90            | degree  | yes           | yes          | yes          | yes          | wind angle (regarding north)                                 |
| al        | 0.8           | N/A     | yes           | no           | no           | no           | albedo                                                       |
| tb        | 0.1           | N/A     | no            | yes          | no           | no           | coefficient for air pollution from 0 (clean) to 1 (polluted) |
| I         | 100           | A       | yes           | yes          | yes          | yes          | transit intensity                                            |
| m         | 1.5           | kg/m    | yes           | yes          | yes          | yes          | mass per unit length (only used in transient mode)           |
| d         | 1.9E-02       | m       | no            | no           | yes          | yes          | core diameter                                                |
| D         | 3.0E-02       | m       | yes           | yes          | yes          | yes          | external (global) diameter                                   |
| a         | 2.84E-04      | m²      | no            | no           | yes          | yes          | core section                                                 |
| A         | 7.07E-04      | m²      | no            | no           | yes          | yes          | external (global) section                                    |
| R         | 4.0E-02       | N/A     | yes           | no           | no           | no           | roughness                                                    |
| l         | 1.0           | W/m/K   | no            | no           | yes          | yes          | radial thermal conductivity                                  |
| c         | 500           | J/kg/K  | yes           | yes          | yes          | yes          | specific heat capacity (only used in transient mode)         |
| alpha     | 0.5           | N/A     | yes           | yes          | yes          | yes          | solar absorption                                             |
| epsilon   | 0.5           | N/A     | yes           | yes          | yes          | yes          | emissivity                                                   |
| RDC20     | 2.5E-05       | Ohm/m   | yes           | no           | yes          | yes          | electric resistance per unit length (DC) at 20°C             |
| km        | 1.006         | N/A     | yes           | no           | yes          | yes          | coefficient for magnetic effects                             |
| ki        | 0.016         | A⁻¹     | no            | no           | yes          | yes          | coefficient for magnetic effects                             |
| kl        | 3.8E-03       | K⁻¹     | yes           | no           | yes          | yes          | linear resistance augmentation with temperature              |
| kq        | 8.0E-07       | K⁻²     | no            | no           | yes          | yes          | quadratic resistance augmentation with temperature           |
| RDCHigh   | 3.05E-05      | Ohm/m   | no            | yes          | no           | no           | electric resistance per unit length (DC) at THigh            |
| RDCLow    | 2.66E-05      | Ohm/m   | no            | yes          | no           | no           | electric resistance per unit length (DC) at TLow             |
| THigh     | 60            | celsius | no            | yes          | no           | no           | temperature for RDCHigh measurement                          |
| TLow      | 20            | celsius | no            | yes          | no           | no           | temperature for RDCHigh measurement                          | 