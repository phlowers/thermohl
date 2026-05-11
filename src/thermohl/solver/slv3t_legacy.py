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
from thermohl.solver.entities import TargetType, CableLocationListLike
from thermohl.solver.slv3t import Solver3T
from thermohl.solver.entities import TemperatureType
from thermohl.solver.solver import (
    temporarily_override_parameter,
    temporarily_override_solar_irradiance,
)


class Solver3TL(Solver3T):
    DERIVATIVE_INCREMENT = 0.1

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

        core_diameter_array = self.args.core_diameter * np.ones(
            (self.args.get_number_of_computations(),)
        )
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

    def morgan_3t(
        self,
        surface_temperature: floatArray,
        core_temperature: floatArray,
        joule_value: Optional[floatArrayLike] = None,
    ) -> floatArray:
        """
        Computes the Morgan function for given temperature arrays.

        Args:
            surface_temperature (numpy.ndarray): Array of surface temperatures.
            core_temperature (numpy.ndarray): Array of core temperatures.
            joule_value (float | numpy.ndarray, optional): Precomputed joule heating value.

        Returns:
            numpy.ndarray: Resulting array after applying the Morgan function.
        """
        if joule_value is None:
            joule_value = self.joule(surface_temperature, core_temperature)

        heat_flux_coefficient = self.morgan_coefficients[0]
        return (
            core_temperature - surface_temperature
        ) - heat_flux_coefficient * joule_value

    def _steady_intensity_header(
        self, T: floatArrayLike, target: CableLocationListLike
    ) -> Tuple[np.ndarray, Callable]:
        """Format input for ampacity solver."""

        max_len = self.args.get_number_of_computations()
        Tmax = T * np.ones(max_len)
        target_ = self._check_target(target, self.args.core_diameter, max_len)

        # pre-compute indexes
        surface_indices = np.nonzero(target_ == TargetType.SURFACE)[0]
        average_indices = np.nonzero(target_ == TargetType.AVERAGE)[0]
        core_indices = np.nonzero(target_ == TargetType.CORE)[0]

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
        offset: floatArray = np.array([]),
        surface_temperature_0: Optional[floatArrayLike] = None,
        core_temperature_0: Optional[floatArrayLike] = None,
        time_constant: float = 600.0,
        return_power: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute transient-state temperature with legacy method.

        Args:
            offset (numpy.ndarray): A 1D array with times (in seconds) when the temperature needs to be
                computed. The array must contain increasing values (undefined behaviour otherwise).
            surface_temperature_0 (float): Initial surface temperature. If set to None, the ambient temperature from
                internal dict will be used. The default is None.
            core_temperature_0 (float): Initial core temperature. If set to None, the ambient temperature from
                internal dict will be used. The default is None.
            time_constant (float): A time-constant to add some inertia. The default is 600.
            return_power (bool, optional): Return power term values. The default is False.

        Returns:
            Dict[str, Any]: A dictionary with temperature and other results (depending on inputs)
                in the keys, along with input data.

        """

        # get sizes (n for input dict entries, N for offsets)
        n = self.args.get_number_of_computations()
        N = len(offset)
        if N < 2:
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

        # inverse of m*C : shortcuts for time-loop
        imc = 1.0 / (self.args.linear_mass * self.args.heat_capacity)

        # define temperature data 2D arrays
        surface_temperature = np.zeros((N, n))
        average_temperature = np.zeros((N, n))
        core_temperature = np.zeros((N, n))
        temperature_difference = np.zeros((N, n))
        # set initial values in first row
        surface_temperature[0, :] = surface_temperature_0
        core_temperature[0, :] = core_temperature_0
        average_temperature[0, :] = self.average(
            surface_temperature[0, :], core_temperature[0, :]
        )
        temperature_difference[0, :] = (
            core_temperature[0, :] - surface_temperature[0, :]
        )

        # compute transient temperatures for each row after the first.
        for i in range(1, len(offset)):
            balance = self.balance_3t(
                surface_temperature[i - 1, :], core_temperature[i - 1, :]
            )
            time_difference = offset[i] - offset[i - 1]
            average_temperature[i, :] = (
                average_temperature[i - 1, :] + time_difference * imc * balance
            )
            temperature_difference[i, :] = temperature_difference[i - 1, :] * (
                1.0 - time_difference / time_constant
            ) + (
                time_difference
                / time_constant
                * self.morgan_coefficients[0]
                * self.joule_heating.value(average_temperature[i, :])
            )
            core_temperature[i, :] = (
                average_temperature[i, :] + 0.5 * temperature_difference[i, :]
            )
            surface_temperature[i, :] = (
                average_temperature[i, :] - 0.5 * temperature_difference[i, :]
            )

        result = self._transient_temperature_results(
            offset,
            surface_temperature,
            average_temperature,
            core_temperature,
            return_power,
            n,
        )

        result = self._add_input_data_to_result(result)

        return result

    def steady_temperature(
        self,
        return_uncertainty=False,
        **kwargs,
    ):
        """
        Compute the steady-state temperature distribution.

        Args:
            surface_temperature_guess (float | numpy.ndarray | None): Initial guess for the surface temperature. If
                None, ambient temperature is used.
            core_temperature_guess (float | numpy.ndarray | None): Initial guess for the core temperature. If None, 1.5
                times the absolute value of ambient temperature is used.
            tol (float): Tolerance for the quasi-Newton solver.
            maxiter (int): Maximum number of iterations for the quasi-Newton solver.
            return_err (bool): If True, the error of the solution is included in the returned dict.
            return_power (bool): If True, power-related values are included in the returned dict.
            return_uncertainty (bool): If True, the uncertainty on computed average temperature is included in the
                returned dict. This is the uncertainty due to uncertainties on transit, ambient temperature, wind speed
                and direction, and solar irradiance. Uncertainties on other parameters (e.g. albedo) are ignored.

        Returns:
            dict[str, np.ndarray]: Dictionary containing the steady-state temperatures and optionally the error and
                power-related values and uncertainty,
                along with input data.
        """
        result = super().steady_temperature(**kwargs)
        if not return_uncertainty:
            return result

        average_temperature = result[TemperatureType.AVERAGE.value]
        uncertainty = self._compute_temperature_uncertainty(
            average_temperature, **kwargs
        )
        result["uncertainty"] = uncertainty

        return result

    def _compute_temperature_uncertainty(
        self,
        temperature: floatArrayLike,
        **kwargs,
    ) -> floatArrayLike:
        uncertainty_transit = 0.05 * self.args["transit"]
        UNCERTAINTY_AMBIENT_TEMPERATURE = 1
        UNCERTAINTY_WIND_SPEED = 1
        UNCERTAINTY_WIND_DIRECTION = 10
        UNCERTAINTY_SOLAR_IRRADIANCE = 100
        self._steady_temperature_partial_derivative(temperature, "transit", **kwargs)
        square_uncertainty = (
            (
                uncertainty_transit
                * self._steady_temperature_partial_derivative(
                    temperature, "transit", **kwargs
                )
            )
            ** 2
            + (
                UNCERTAINTY_AMBIENT_TEMPERATURE
                * self._steady_temperature_partial_derivative(
                    temperature,
                    "ambient_temperature",
                    **kwargs,
                )
            )
            ** 2
            + (
                UNCERTAINTY_WIND_SPEED
                * self._steady_temperature_partial_derivative(
                    temperature, "wind_speed", **kwargs
                )
            )
            ** 2
            + (
                UNCERTAINTY_WIND_DIRECTION
                * self._steady_temperature_partial_derivative(
                    temperature,
                    "wind_azimuth",
                    **kwargs,
                )
            )
            ** 2
            + (
                UNCERTAINTY_SOLAR_IRRADIANCE
                * self._steady_temperature_partial_derivative_irradiance(
                    temperature, **kwargs
                )
            )
            ** 2
        )
        return square_uncertainty**0.5

    def _steady_temperature_partial_derivative(
        self,
        temperature: floatArrayLike,
        parameter_name: str,
        **kwargs,
    ) -> floatArrayLike:
        try:
            incremented_parameter_value = (
                self.args.__getattribute__(parameter_name) + self.DERIVATIVE_INCREMENT
            )
            with temporarily_override_parameter(
                self, parameter_name, incremented_parameter_value
            ):
                kwargs.update(
                    {
                        "surface_temperature_guess": temperature,
                        "core_temperature_guess": temperature,
                        "return_err": False,
                        "return_power": False,
                        "return_uncertainty": False,
                    }
                )
                temperature_bis = self.steady_temperature(
                    **kwargs,
                )[TemperatureType.AVERAGE.value]
        except AttributeError:
            raise ValueError(
                f"Solver.args doesn't include {parameter_name}, can't compute partial derivative"
            )
        return self._approximate_derivative(temperature_bis, temperature)

    def _steady_temperature_partial_derivative_irradiance(
        self,
        temperature: floatArrayLike,
        **kwargs,
    ):
        incremented_solar_irradiance = (
            self.solar_heating.solar_irradiance + self.DERIVATIVE_INCREMENT
        )
        with temporarily_override_solar_irradiance(self, incremented_solar_irradiance):
            kwargs.update(
                {
                    "surface_temperature_guess": temperature,
                    "core_temperature_guess": temperature,
                    "return_err": False,
                    "return_power": False,
                    "return_uncertainty": False,
                }
            )
            temperature_with_increased_solar_irradiance = self.steady_temperature(
                **kwargs,
            )[TemperatureType.AVERAGE.value]

        return self._approximate_derivative(
            temperature_with_increased_solar_irradiance, temperature
        )

    @classmethod
    def _approximate_derivative(cls, value_a, value_b) -> floatArrayLike:
        return (value_a - value_b) / cls.DERIVATIVE_INCREMENT
