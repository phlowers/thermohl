# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import numbers
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from thermohl import floatArrayLike, floatArray
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
    ) -> pd.DataFrame:
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
            pandas.DataFrame: A DataFrame with temperature and other results (depending on inputs) in the columns.

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
        df = pd.DataFrame(
            data=conductor_temperature, columns=[VariableType.TEMPERATURE]
        )

        if return_err:
            df[VariableType.ERROR] = err

        if return_power:
            df[PowerType.JOULE] = self.joule_heating.value(conductor_temperature)
            df[PowerType.SOLAR] = self.solar_heating.value(conductor_temperature)
            df[PowerType.CONVECTION] = self.convective_cooling.value(
                conductor_temperature
            )
            df[PowerType.RADIATION] = self.radiative_cooling.value(
                conductor_temperature
            )
            df[PowerType.RAIN] = self.precipitation_cooling.value(conductor_temperature)

        return df

    def transient_temperature(
        self,
        offset: floatArray = np.array([]),
        T0: Optional[float] = None,
        return_power: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute transient-state temperature.

        Args:
            offset (numpy.ndarray): A 1D array with times (in seconds) when the temperature needs to be computed. The array must contain increasing values (undefined behaviour otherwise).
            T0 (float | None): Initial temperature. If None, the ambient temperature from the internal dict will be used. The default is None.
            return_power (bool, optional): Return power term values. The default is False.

        Returns:
            Dict[str, Any]: A dictionary with temperature and other results (depending on inputs) in the keys.
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
            VariableType.TIME: offset,
            VariableType.TEMPERATURE: conductor_temperature,
        }

        # manage return dict 2: powers
        if return_power:
            for power in Solver_.powers():
                result[power] = np.zeros_like(conductor_temperature)
            for i in range(N):
                for key in time_changing_parameters.keys():
                    self.args[key] = time_changing_parameters[key][i, :]
                self.update()
                result[PowerType.JOULE][i, :] = self.joule_heating.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.SOLAR][i, :] = self.solar_heating.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.CONVECTION][i, :] = self.convective_cooling.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.RADIATION][i, :] = self.radiative_cooling.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.RAIN][i, :] = self.precipitation_cooling.value(
                    conductor_temperature[i, :]
                )

        # squeeze return values if n is 1
        if n == 1:
            keys = list(result.keys())
            keys.remove(VariableType.TIME)
            for key in keys:
                result[key] = result[key][:, 0]

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
    ) -> pd.DataFrame:
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
            pandas.DataFrame: A dataframe with maximum intensity and other results (depending on inputs) in the columns.

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
        df = pd.DataFrame(data=A, columns=[VariableType.TRANSIT])

        if return_err:
            df[VariableType.ERROR] = err

        if return_power:
            df[PowerType.JOULE] = self.joule_heating.value(max_conductor_temperature)
            df[PowerType.SOLAR] = self.solar_heating.value(max_conductor_temperature)
            df[PowerType.CONVECTION] = self.convective_cooling.value(
                max_conductor_temperature
            )
            df[PowerType.RADIATION] = self.radiative_cooling.value(
                max_conductor_temperature
            )
            df[PowerType.RAIN] = self.precipitation_cooling.value(
                max_conductor_temperature
            )

        return df

    def _set_default_reduced_intensity_args(
        self,
        ambient_temperature: Optional[floatArrayLike],
        wind_speed: Optional[floatArrayLike],
        measured_global_radiation: Optional[floatArrayLike],
    ):
        if ambient_temperature is None:
            logger.warning(
                "ambient_temperature is not set. Using default value of 30 °C."
            )
            ambient_temperature = 30.0
        self.args.ambient_temperature = ambient_temperature

        if wind_speed is None:
            logger.warning("wind_speed is not set. Using default value of 0.6 m/s.")
            wind_speed = 0.6
        self.args.wind_speed = wind_speed

        if measured_global_radiation is None:
            logger.warning(
                "measured_global_radiation is not set. Using default value of 600 W/m²."
            )
            measured_global_radiation = 600.0
        self.args.measured_global_radiation = measured_global_radiation

    def reduced_intensity(
        self,
        measured_temperature_difference: floatArrayLike,
        measured_intensity: floatArrayLike,
        ambient_temperature: Optional[floatArrayLike] = None,
        wind_speed: Optional[floatArrayLike] = None,
        measured_global_radiation: Optional[floatArrayLike] = None,
        max_conductor_temperature: Optional[floatArrayLike] = None,
    ) -> floatArrayLike:
        """
        Compute the reduced intensity limit for a given measured temperature difference
        betwwen the sound cable and a hotspot on the junction between a cable
        and a faulty sleeve.

        Args:
            measured_temperature_difference (float | np.ndarray): The measured temperature difference between the cable surface and the sleeve.
            measured_intensity (float | np.ndarray): The measuredintensity at which the temperature difference was measured.
            ambient_temperature (Optional[float | np.ndarray]): The ambient temperature. Default is 30.
            wind_speed (Optional[float | np.ndarray]): The wind speed. Default is 0.6.
            measured_global_radiation (Optional[float | np.ndarray]): The measured solar irradiance. Default is 600.
            max_conductor_temperature (Optional[float | np.ndarray]): The maximum conductor temperature. Default is 100.
        """
        # Save args that will be modified so as to be able to restore them at the end of the computation
        solver_transit = self.args.transit
        solver_ambient_temperature = self.args.ambient_temperature
        solver_wind_speed = self.args.wind_speed
        solver_measured_solar_irradiance = self.args.measured_global_radiation
        solver_wind_attack_angle = self.args.wind_attack_angle

        # Set args default values for reduced intensity computation.
        # These differ from those used for the other computations.
        self._set_default_reduced_intensity_args(
            ambient_temperature, wind_speed, measured_global_radiation
        )

        # Set default value for max_conductor_temperature if not provided.
        if max_conductor_temperature is None:
            max_conductor_temperature = np.full_like(measured_intensity, 100.0)

        self.args.wind_attack_angle = 90.0
        self.convective_cooling.__init__(**self.args.__dict__)

        def conductor_temperature(transit):
            self.args.transit = transit
            self.joule_heating.__init__(**self.args.__dict__)
            return self.steady_temperature()[VariableType.TEMPERATURE][0]

        def temperature_difference(transit):
            return measured_temperature_difference * (
                (transit / measured_intensity) ** 2
            )

        def sleeve_temperature(transit):
            return conductor_temperature(transit) + temperature_difference(transit)

        def f(transit):
            return sleeve_temperature(transit) - max_conductor_temperature

        x0 = np.full_like(measured_intensity, 100.0)

        reduced_intensity = quasi_newton(f, x0=x0)

        # Restore previous args
        self.args.transit = solver_transit
        # Update joule heating with restored transit
        self.joule_heating.__init__(**self.args.__dict__)
        self.args.ambient_temperature = solver_ambient_temperature
        self.args.wind_speed = solver_wind_speed
        self.args.measured_global_radiation = solver_measured_solar_irradiance
        self.args.wind_attack_angle = solver_wind_attack_angle
        # Update convective cooling with restored wind_attack_angle
        self.convective_cooling.__init__(**self.args.__dict__)

        return reduced_intensity
