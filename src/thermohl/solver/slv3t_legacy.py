# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple, Type, Optional, Any, Callable, Final, Dict

import numpy as np

from thermohl import floatArrayLike, floatArray, intArray
from thermohl.power import PowerTerm
from thermohl.solver.enums.cable_location import CableLocation, CableLocationListLike
from thermohl.solver.slv3t import Solver3T


class Solver3TL(Solver3T):
    def __init__(
        self,
        dic: Optional[dict[str, Any]] = None,
        joule: Type[PowerTerm] = PowerTerm,
        solar: Type[PowerTerm] = PowerTerm,
        convective: Type[PowerTerm] = PowerTerm,
        radiative: Type[PowerTerm] = PowerTerm,
        precipitation: Type[PowerTerm] = PowerTerm,
    ):
        super().__init__(dic, joule, solar, convective, radiative, precipitation)
        self.update()

    def _morgan_coefficients(self) -> Tuple[floatArray, intArray]:
        """
        Calculate coefficients for heat flux between surface and core in steady state.

        Returns:
            Tuple[numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[int]]
                - heat_flux_coefficients : numpy.ndarray[float]
                    Coefficient array for heat flux.
                - indices_non_zero_diameter : numpy.ndarray[int]
                    Indices where core diameter is greater than 0.
                    When conductors are uniform, core diameter is equal to 0.0.
                    When conductors are bimetallic, core diameter is greater than 0.0.
        """
        UNIFORM_CONDUCTOR_COEFFICIENT: Final[float] = 1 / 13
        BIMETALLIC_CONDUCTOR_COEFFICIENT: Final[float] = 1 / 21

        core_diameter_array = self.args.core_diameter * np.ones((self.args.max_len(),))
        indices_non_zero_diameter = np.nonzero(core_diameter_array > 0.0)[0]
        heat_flux_coefficients = UNIFORM_CONDUCTOR_COEFFICIENT * np.ones_like(
            core_diameter_array
        )
        heat_flux_coefficients[indices_non_zero_diameter] = (
            BIMETALLIC_CONDUCTOR_COEFFICIENT
        )
        return heat_flux_coefficients, indices_non_zero_diameter

    def average(self, surface_temperature, core_temperature):
        """
        Compute average temperature given surface and core temperature.

        Unlike Solver3T, always use a regular mean even for non-homogeneous
        conductors.

        Args:
            surface_temperature (numpy.ndarray): Array of surface temperatures.
            core_temperature (numpy.ndarray): Array of core temperatures.
        """
        return 0.5 * (surface_temperature + core_temperature)

    def morgan(
        self, surface_temperature: floatArray, core_temperature: floatArray
    ) -> floatArray:
        """
        Computes the Morgan function for given temperature arrays.

        Args:
            surface_temperature (numpy.ndarray): Array of surface temperatures.
            core_temperature (numpy.ndarray): Array of core temperatures.

        Returns:
            numpy.ndarray: Resulting array after applying the Morgan function.
        """
        morgan_coefficient = self.morgan_coefficients[0]
        return (
            core_temperature - surface_temperature
        ) - morgan_coefficient * self.joule(surface_temperature, core_temperature)

    def _steady_intensity_header(
        self, T: floatArrayLike, target: CableLocationListLike
    ) -> Tuple[np.ndarray, Callable]:
        """Format input for ampacity solver."""

        max_len = self.args.max_len()
        Tmax = T * np.ones(max_len)
        target_ = self._check_target(target, self.args.core_diameter, max_len)

        # pre-compute indexes
        surface_indices = np.nonzero(target_ == CableLocation.SURFACE)[0]
        average_indices = np.nonzero(target_ == CableLocation.AVERAGE)[0]
        core_indices = np.nonzero(target_ == CableLocation.CORE)[0]

        def newtheader(
            transit: floatArray, tg: floatArray
        ) -> Tuple[floatArray, floatArray]:
            self.args.transit = transit
            self.joule_heating.__init__(**self.args.__dict__)
            surface_temperature = np.ones_like(tg) * np.nan
            core_temperature = np.ones_like(tg) * np.nan

            surface_temperature[surface_indices] = Tmax[surface_indices]
            core_temperature[surface_indices] = tg[surface_indices]

            surface_temperature[average_indices] = tg[average_indices]
            core_temperature[average_indices] = (
                2 * Tmax[average_indices] - surface_temperature[average_indices]
            )

            core_temperature[core_indices] = Tmax[core_indices]
            surface_temperature[core_indices] = tg[core_indices]

            return surface_temperature, core_temperature

        return Tmax, newtheader

    def _morgan_transient(self):
        """Morgan coefficients for transient temperature."""
        morgan_coeff_1, _ = self.morgan_coefficients
        morgan_coeff_2 = 0.5 * np.ones_like(morgan_coeff_1)
        return morgan_coeff_1, morgan_coeff_2

    def transient_temperature_legacy(
        self,
        time: floatArray = np.array([]),
        surface_temperature_0: Optional[floatArrayLike] = None,
        core_temperature_0: Optional[floatArrayLike] = None,
        time_constant: float = 600.0,
        return_power: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute transient-state temperature with legacy method.

        Args:
            time (numpy.ndarray): A 1D array with times (in seconds) when the temperature needs to be
                computed. The array must contain increasing values (undefined behaviour otherwise).
            surface_temperature_0 (float): Initial surface temperature. If set to None, the ambient temperature from
                internal dict will be used. The default is None.
            core_temperature_0 (float): Initial core temperature. If set to None, the ambient temperature from
                internal dict will be used. The default is None.
            time_constant (float): A time-constant to add some inertia. The default is 600.
            return_power (bool, optional): Return power term values. The default is False.

        Returns:
            Dict[str, Any]: A dictionary with temperature and other results (depending on inputs)
                in the keys.

        """

        # get sizes (input_size for input dict entries, time_size for time)
        input_size = self.args.max_len()
        time_size = len(time)
        if time_size < 2:
            raise ValueError()

        # get initial temperature
        surface_temperature_0 = (
            surface_temperature_0
            if surface_temperature_0 is not None
            else self.args.ambient_temperature
        )
        core_temperature_0 = (
            core_temperature_0
            if core_temperature_0 is not None
            else 1.0 + surface_temperature_0
        )

        # shortcuts for time-loop
        imc = 1.0 / (self.args.linear_mass * self.args.heat_capacity)

        # init
        surface_temperature = np.zeros((time_size, input_size))
        ambient_temperature = np.zeros((time_size, input_size))
        core_temperature = np.zeros((time_size, input_size))
        temperature_difference_c = np.zeros((time_size, input_size))

        surface_temperature[0, :] = surface_temperature_0
        core_temperature[0, :] = core_temperature_0
        ambient_temperature[0, :] = self.average(
            surface_temperature[0, :], core_temperature[0, :]
        )
        temperature_difference_c[0, :] = (
            core_temperature[0, :] - surface_temperature[0, :]
        )

        for i in range(1, len(time)):
            balance = self.balance(
                surface_temperature[i - 1, :], core_temperature[i - 1, :]
            )
            time_difference = time[i] - time[i - 1]
            ambient_temperature[i, :] = (
                ambient_temperature[i - 1, :] + time_difference * imc * balance
            )
            temperature_difference_c[i, :] = (
                1.0 - time_difference / time_constant
            ) * temperature_difference_c[i - 1, :] + (
                time_difference
                / time_constant
                * self.morgan_coefficients[0]
                * self.joule_heating.value(ambient_temperature[i, :])
            )
            core_temperature[i, :] = (
                ambient_temperature[i, :] + 0.5 * temperature_difference_c[i, :]
            )
            surface_temperature[i, :] = (
                ambient_temperature[i, :] - 0.5 * temperature_difference_c[i, :]
            )

        return self._transient_temperature_results(
            time,
            surface_temperature,
            ambient_temperature,
            core_temperature,
            return_power,
            input_size,
        )
