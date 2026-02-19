# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Thermodynamics quantities. Values from Wikipedia unless specified."""

import numpy as np
from thermohl import floatArrayLike

# Standard temperature and pressure from EPA and NIST; in K and Pa
_STD_TEMP_K, _STD_PRESSURE_PA = 293.15, 1.01325e05

# Boltzmann constant, Avogadro number and Gas constant (all in SI)
_BOLTZMANN_CONSTANT = 1.380649e-23
_AVOGADRO_NUMBER = 6.02214076e23
_GAS_CONSTANT = _BOLTZMANN_CONSTANT * _AVOGADRO_NUMBER


class Air:
    @staticmethod
    def heat_capacity(temperature: floatArrayLike = _STD_TEMP_K) -> floatArrayLike:
        """In J.kg**-1.K**-1"""
        return np.interp(temperature, [240.0, 600.0], [1.006, 1.051])


class Water:
    @staticmethod
    def boiling_point(pressure: floatArrayLike = _STD_PRESSURE_PA) -> floatArrayLike:
        """Using Clausius–Clapeyron equation; in K."""
        # convert H from J.kg**-1 to J.mol**-1 using molar mass
        latent_heat = Water.heat_of_vaporization() * 0.018015
        boiling_temp = 1.0 / (
            1 / 373.15
            - _GAS_CONSTANT * np.log(pressure / _STD_PRESSURE_PA) / latent_heat
        )
        return boiling_temp

    @staticmethod
    def heat_capacity(temperature: floatArrayLike = _STD_TEMP_K) -> floatArrayLike:
        """From NIST webbook; in J.kg**-1.K**-1.
        See https://webbook.nist.gov/cgi/cbook.cgi?Name=Water&Units=SI.
        """
        coeff_a = -203.6060
        coeff_b = +1523.290
        coeff_c = -3196.413
        coeff_d = +2474.455
        coeff_e = 3.855326
        temp_kilo = temperature / 1000.0
        return (
            coeff_a
            + coeff_b * temp_kilo
            + coeff_c * temp_kilo**2
            + coeff_d * temp_kilo**3
            + coeff_e / temp_kilo**2
        )

    @staticmethod
    def heat_of_vaporization() -> float:
        # """At T=373.15 K and normal pressure; in J.mol**-1."""
        # return 4.0660E+04
        """At T=373.15 K and normal pressure; in J.kg**-1."""
        return 2.257e05

    @staticmethod
    def vapor_pressure(temperature: floatArrayLike = _STD_TEMP_K) -> floatArrayLike:
        """Using Buck equation; in Pa."""
        temperature = temperature - 273.15
        return 611.21 * np.exp(
            (18.678 - temperature / 234.5) * (temperature / (257.14 + temperature))
        )

    @staticmethod
    def volumic_mass(temperature: floatArrayLike = _STD_TEMP_K) -> floatArrayLike:
        """In kg.m**-3."""
        temp_points = np.array(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
                25.0,
                26.0,
                27.0,
                28.0,
                29.0,
                30.0,
                31.0,
                32.0,
                33.0,
                34.0,
                35.0,
                36.0,
                37.0,
                38.0,
                39.0,
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
                65.0,
                70.0,
                75.0,
                80.0,
                85.0,
                90.0,
                95.0,
                100.0,
            ]
        )
        density_points = np.array(
            [
                999.840,
                999.899,
                999.940,
                999.964,
                999.972,
                999.964,
                999.940,
                999.902,
                999.848,
                999.781,
                999.700,
                999.6,
                999.5,
                999.38,
                999.24,
                999.1,
                998.94,
                998.77,
                998.59,
                998.4,
                998.2,
                997.99,
                997.77,
                997.54,
                997.29,
                997.04,
                996.78,
                996.51,
                996.23,
                995.94,
                995.64,
                995.34,
                995.02,
                994.7,
                994.37,
                994.03,
                993.68,
                993.32,
                992.96,
                992.59,
                992.21,
                990.21,
                988.03,
                985.69,
                983.19,
                980.55,
                977.76,
                974.84,
                971.79,
                968.61,
                965.3,
                961.88,
                958.35,
            ]
        )
        return np.interp(temperature, temp_points, density_points)


class Ice:
    @staticmethod
    def heat_capacity() -> float:
        """In J.kg**-1.K**-1"""
        return 2.093e03

    @staticmethod
    def heat_of_fusion() -> float:
        """At T=273.15 K and normal pressure; in J.kg**-1."""
        return 3.3355e05
