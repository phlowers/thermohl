# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import numbers
from typing import Optional

import numpy as np

from thermohl import floatArrayLike, floatArray
from thermohl.power.solar_heating import FixedSolarIrradianceSolarHeating
from thermohl.solver.solver import Solver as Solver_, get_time_changing_parameters
from thermohl.solver.parameters import DEFAULT_PARAMETERS as default
from thermohl.solver.entities import PowerType, VariableType
from thermohl.utils import bisect_v
from thermohl.utils import quasi_newton


logger = logging.getLogger(__name__)


class Solver1T(Solver_):
    def steady_temperature(
        self,
        Tmin: float = default.tmin,
        Tmax: float = default.tmax,
        tol: float = default.tol,
        maxiter: int = default.maxiter,
        return_err: bool = False,
        return_power: bool = True,
    ) -> dict[str, np.array]:
        """
        Compute steady-state temperature.

        Args:
            Tmin (float, optional): Lower bound for temperature.
            Tmax (float, optional): Upper bound for temperature.
            tol (float, optional): Tolerance for temperature error.
            maxiter (int, optional): Max number of iterations.
            return_err (bool, optional): Return final error on temperature to check convergence. The default is False.
            return_power (bool, optional): Return power term values. The default is True.

        Returns:
            dict[str, np.array]: A dictionary with temperature and other results (depending on inputs) in the keys,
            along with input data.

        """

        # solve with bisection
        conductor_temperature, err = bisect_v(
            lambda x: -self.balance(x),
            Tmin,
            Tmax,
            (self.args.get_number_of_computations(),),
            tol,
            maxiter,
        )

        # format output
        result = {
            VariableType.TEMPERATURE.value: conductor_temperature,
        }
        self.add_error_and_power_if_needed(
            conductor_temperature, err, result, return_err, return_power
        )
        result = self._add_input_data_to_result(result)
        return result

    def transient_temperature(
        self,
        offset: floatArray = np.array([]),
        T0: Optional[float] = None,
        return_power: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Compute transient-state temperature.

        Args:
            offset (numpy.ndarray): A 1D array with times (in seconds) when the temperature needs to be computed. The array must contain increasing values (undefined behaviour otherwise).
            T0 (float | None): Initial temperature. If None, the ambient temperature from the internal dict will be used. The default is None.
            return_power (bool, optional): Return power term values. The default is False.

        Returns:
            dict[str, np.ndarray]: A dictionary with temperature and other results (depending on inputs) in the keys, along with input data.
        """

        # get sizes
        n = self.args.get_number_of_computations()
        N = len(offset)
        if N < 2:
            raise ValueError("The length of the time array must be at least 2.")

        # get initial temperature
        if T0 is None:
            T0 = (
                self.args.ambient_temperature
                if isinstance(self.args.ambient_temperature, numbers.Number)
                else self.args.ambient_temperature[0]
            )
        time_changing_parameters = get_time_changing_parameters(self.args, offset, N, n)
        # inverse of m*C : shortcuts for time-loop
        imc = 1.0 / (self.args.linear_mass * self.args.heat_capacity)

        # init
        conductor_temperature = np.zeros((N, n))
        conductor_temperature[0, :] = T0

        # main time loop
        for i in range(1, N):
            for k, v in time_changing_parameters.items():
                self.args[k] = v[i, :]
            self.update()
            conductor_temperature[i, :] = (
                conductor_temperature[i - 1, :]
                + (offset[i] - offset[i - 1])
                * self.balance(conductor_temperature[i - 1, :])
                * imc
            )

        # save results
        result = {
            VariableType.TIME.value: offset,
            VariableType.TEMPERATURE.value: conductor_temperature,
        }

        # manage return dict 2: powers
        if return_power:
            for power in Solver_.powers():
                result[power.value] = np.zeros_like(conductor_temperature)
            for i in range(N):
                for key in time_changing_parameters.keys():
                    self.args[key] = time_changing_parameters[key][i, :]
                self.update()
                result[PowerType.JOULE.value][i, :] = self.joule_heating.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.SOLAR.value][i, :] = self.solar_heating.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.CONVECTION.value][i, :] = (
                    self.convective_cooling.value(conductor_temperature[i, :])
                )
                result[PowerType.RADIATION.value][i, :] = self.radiative_cooling.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.RAIN.value][i, :] = self.precipitation_cooling.value(
                    conductor_temperature[i, :]
                )

        # squeeze return values if n is 1
        if n == 1:
            keys = list(result.keys())
            keys.remove(VariableType.TIME.value)
            for key in keys:
                result[key] = result[key][:, 0]

        result = self._add_input_data_to_result(result)

        return result

    def steady_intensity(
        self,
        max_conductor_temperature: floatArrayLike = np.array([]),
        Imin: float = default.imin,
        Imax: float = default.imax,
        tol: float = default.tol,
        maxiter: int = default.maxiter,
        return_err: bool = False,
        return_power: bool = True,
    ) -> dict[str, np.ndarray]:
        """Compute steady-state max intensity.

        Compute the maximum intensity that can be run in a conductor without
        exceeding the temperature given in argument.

        Args:
            max_conductor_temperature (float | numpy.ndarray): Maximum temperature.
            Imin (float, optional): Lower bound for intensity. The default is 0.
            Imax (float, optional): Upper bound for intensity. The default is 9999.
            tol (float, optional): Tolerance for temperature error. The default is 1.0E-06.
            maxiter (int, optional): Max number of iterations. The default is 64.
            return_err (bool, optional): Return final error on intensity to check convergence. The default is False.
            return_power (bool, optional): Return power term values. The default is True.

        Returns:
            dict[str, np.ndarray]: A dictionary with maximum intensity and other results (depending on inputs) in the keys,
            along with input data.

        """

        # save transit in arg
        transit = self.args.transit

        # solve with bisection
        shape = (self.args.get_number_of_computations(),)
        T_ = max_conductor_temperature * np.ones(shape)
        joule_heating = (
            self.convective_cooling.value(T_)
            + self.radiative_cooling.value(T_)
            + self.precipitation_cooling.value(T_)
            - self.solar_heating.value(T_)
        )

        def fun(i: floatArray) -> floatArrayLike:
            self.args.transit = i
            self.joule_heating.__init__(**self.args.__dict__)
            return self.joule_heating.value(T_) - joule_heating

        A, err = bisect_v(fun, Imin, Imax, shape, tol, maxiter)

        # restore previous transit
        self.args.transit = transit

        # format output
        result = {VariableType.TRANSIT.value: A}

        self.add_error_and_power_if_needed(
            max_conductor_temperature,
            err,
            result,
            return_err,
            return_power,
        )

        result = self._add_input_data_to_result(result)
        return result

    def reduced_intensity(
        self,
        measured_temperature_difference: float,
        measured_intensity: float,
        ambient_temperature: float = 30.0,
        wind_speed: float = 0.6,
        solar_irradiance: float = 600.0,
        max_conductor_temperature: float = 100.0,
    ) -> float:
        """
        Compute the reduced intensity limit for a given measured temperature difference
        between the sound cable and a hotspot on the junction between a cable
        and a faulty sleeve.

        Args:
            measured_temperature_difference (float): The measured temperature difference between the cable surface and the sleeve.
            measured_intensity (float): The measured intensity at which the temperature difference was measured.
            ambient_temperature (Optional[float]): The ambient temperature. Default is 30.
            wind_speed (Optional[float]): The wind speed (more precisely, speed of the wind component perpendicular to the cable). Default is 0.6.
            solar_irradiance (Optional[float]): The measured solar irradiance. Default is 600.
            max_conductor_temperature (Optional[float]): The maximum conductor temperature. Default is 100.

        NB: Default values for optional parameters differ from the default values used for
        other computations.
        """
        with self.temporarily_override_parameters(
            ambient_temperature=ambient_temperature,
            wind_speed=wind_speed,
            wind_attack_angle=np.pi / 2,
        ):
            try:
                saved_solar_heating = self.solar_heating
                self.args.fixed_solar_irradiance = solar_irradiance

                self.solar_heating = FixedSolarIrradianceSolarHeating(
                    **self.args.__dict__,
                )

                def conductor_temperature(transit):
                    with self.temporarily_override_parameters(
                        transit=transit,
                    ):
                        return self.steady_temperature()[
                            VariableType.TEMPERATURE.value
                        ][0]

                def temperature_difference(transit):
                    return measured_temperature_difference * (
                        (transit / measured_intensity) ** 2
                    )

                def sleeve_temperature(transit):
                    return conductor_temperature(transit) + temperature_difference(
                        transit
                    )

                def f(transit):
                    return sleeve_temperature(transit) - max_conductor_temperature

                reduced_intensity = quasi_newton(f, x0=100)

            finally:
                self.solar_heating = saved_solar_heating
                del self.args.fixed_solar_irradiance

        return reduced_intensity
