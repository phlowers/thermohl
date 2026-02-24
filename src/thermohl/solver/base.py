# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Base class to build a solver for heat equation."""

import datetime
from abc import ABC, abstractmethod
from typing import Tuple, Type, Any, Optional, KeysView, Dict

import numpy as np
import pandas as pd
from numpy import ndarray

from thermohl import (
    floatArrayLike,
    floatArray,
    intArray,
    numberArray,
    numberArrayLike,
)
from thermohl.power import PowerTerm
from thermohl.solver.enums.power_type import PowerType


class _DEFPARAM:
    tmin = -99.0
    tmax = +999.0
    tol = 1.0e-09
    maxiter = 64
    imin = 0.0
    imax = 9999.0


class Args:
    """Object to store Solver args in a dict-like manner."""

    # __slots__ = [
    #     'latitude', 'longitude', 'altitude', 'azimuth', 'month', 'day', 'hour', 'ambient_temperature', 'ambient_pressure', 'relative_humidity', 'precipitation_rate', 'wind_speed', 'wind_angle', 'albedo', 'turbidity', 'transit', 'm',
    #     'core_diameter', 'outer_diameter', 'core_area', 'outer_area', 'roughness_ratio', 'radial_thermal_conductivity', 'heat_capacity', 'solar_absorptivity', 'emissivity', 'linear_resistance_dc_20c', 'magnetic_coeff', 'magnetic_coeff_per_a', 'temperature_coeff_linear', 'temperature_coeff_quadratic', 'linear_resistance_temp_high', 'linear_resistance_temp_low',
    #     'temp_high', 'temp_low'
    # ]

    def __init__(self, input_dict: Optional[dict[str, Any]] = None):
        # add default values
        self._set_default_values()
        # use values from input dict
        if input_dict is None:
            input_dict = {}
        keys = self.keys()
        for key in input_dict:
            if key in keys and input_dict[key] is not None:
                self[key] = input_dict[key]

    def _set_default_values(self) -> None:
        """Set default values."""

        self.measured_solar_irradiance = np.nan  # solar irradiance
        self.latitude = 45.0  # latitude (deg)
        self.longitude = 0.0  # longitude (deg)
        self.altitude = 0.0  # altitude (m)
        self.azimuth = 0.0  # azimuth (deg)

        self.month = 3  # month number (1=Jan, 2=Feb, ...)
        self.day = 21  # day of the month
        self.hour = 12  # hour of the day (in [0, 23] range)

        self.ambient_temperature = 15.0  # ambient temperature (C)
        self.ambient_pressure = 1.0e05  # ambient pressure (Pa)
        self.relative_humidity = 0.8  # relative humidity (none, in [0, 1])
        self.precipitation_rate = 0.0  # rain precipitation rate (m.s**-1)
        self.wind_speed = 0.0  # wind speed (m.s**-1)
        self.wind_angle = 90.0  # wind angle (deg, regarding north)
        self.albedo = 0.15  # albedo (1)
        self.nebulosity = 0
        # coefficient for air pollution from 0 (clean) to 1 (polluted)
        self.turbidity = 0.1

        self.transit = 100.0  # transit intensity (A)

        self.linear_mass = 1.5  # mass per unit length (kg.m**-1)
        self.core_diameter = 1.9e-02  # core diameter (m)
        self.outer_diameter = 3.0e-02  # external (global) diameter (m)
        self.core_area = 2.84e-04  # core section (m**2)
        self.outer_area = 7.07e-04  # external (global) section (m**2)
        self.roughness_ratio = 4.0e-02  # roughness (1)
        self.radial_thermal_conductivity = (
            1.0  # radial thermal conductivity (W.m**-1.K**-1)
        )
        self.heat_capacity = 500.0  # specific heat capacity (J.kg**-1.K**-1)

        self.solar_absorptivity = 0.5  # solar absorption (1)
        self.emissivity = 0.5  # emissivity (1)
        # electric resistance per unit length (DC) at 20°C (Ohm.m**-1)
        self.linear_resistance_dc_20c = 2.5e-05

        self.magnetic_coeff = 1.006  # coefficient for magnetic effects (1)
        self.magnetic_coeff_per_a = 0.016  # coefficient for magnetic effects (A**-1)
        # linear resistance augmentation with temperature (K**-1)
        self.temperature_coeff_linear = 3.8e-03
        # quadratic resistance augmentation with temperature (K**-2)
        self.temperature_coeff_quadratic = 8.0e-07
        # electric resistance per unit length (DC) at temp_high (Ohm.m**-1)
        self.linear_resistance_temp_high = 3.05e-05
        # electric resistance per unit length (DC) at temp_low (Ohm.m**-1)
        self.linear_resistance_temp_low = 2.66e-05
        self.temp_high = (
            60.0  # temperature for linear_resistance_temp_high measurement (°C)
        )
        self.temp_low = (
            20.0  # temperature for linear_resistance_temp_low measurement (°C)
        )

    def keys(self) -> KeysView[str]:
        """Get list of members as dict keys."""
        return self.__dict__.keys()

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def max_len(self) -> int:
        """
        Calculate the maximum length of the values in the dictionary.

        This method iterates over all keys in the dictionary and determines the maximum length
        of the values associated with those keys.
        If a value is not of a type that has a length, it is ignored.

        Returns:
            int: The maximum length of the values in the dictionary. If the dictionary is empty
            or all values are of types that do not have a length, the method returns 1.
        """
        result = 1
        for k in self.keys():
            try:
                result = max(result, len(self[k]))
            except TypeError:
                pass
        return result

    def extend_to_max_len(self) -> None:
        """
        Extend all elements in the dictionary to the maximum length.

        This method iterates over all keys in the dictionary and checks if the
        corresponding value is a numpy ndarray. If it is, it checks if its length
        matches the maximum length obtained from the `max_len` method.
        If the length matches, it creates a copy of the array.
        If the length does not match or for non-ndarray values, it creates
        a new numpy array of the maximum length, filled with the original value
        and having the same data type.

        Returns:
            None
        """
        max_len = self.max_len()
        for k in self.keys():
            if isinstance(self[k], np.ndarray):
                t = self[k].dtype
                c = len(self[k]) == max_len
            else:
                t = type(self[k])
                c = False
            if c:
                self[k] = self[k][:]
            else:
                self[k] = self[k] * np.ones((max_len,), dtype=t)

    def compress(self) -> None:
        """
        Compresses the values in the dictionary by replacing numpy arrays with a
        single unique value if all elements in the array are the same.

        Returns:
            None
        """
        for key in self.keys():
            if isinstance(self[key], np.ndarray):
                u = np.unique(self[key])
                if len(u) == 1:
                    self[key] = u[0]


class Solver(ABC):
    """Object to solve a temperature problem.

    The temperature of a conductor is driven by four power terms, two heating
    terms (joule and solar heating) and three cooling terms (convective,
    radiative and precipitation cooling). This class is used to solve a
    temperature problem with the heating and cooling terms passed to its
    __init__ function.
    """

    def __init__(
        self,
        dic: Optional[dict[str, Any]] = None,
        joule: Type[PowerTerm] = PowerTerm,
        solar: Type[PowerTerm] = PowerTerm,
        convective: Type[PowerTerm] = PowerTerm,
        radiative: Type[PowerTerm] = PowerTerm,
        precipitation: Type[PowerTerm] = PowerTerm,
    ) -> None:
        """Create a Solver object.

        Args:
            dic (dict[str, Any] | None): Input values used in power terms. If there is a missing value, a default is used.
            joule (Type[PowerTerm]): Joule heating term class.
            solar (Type[PowerTerm]): Solar heating term class.
            convective (Type[PowerTerm]): Convective cooling term class.
            radiative (Type[PowerTerm]): Radiative cooling term class.
            precipitation (Type[PowerTerm]): Precipitation cooling term class.

        Returns:
            None

        """
        self.args = Args(dic)
        self.args.extend_to_max_len()
        self.joule_heating = joule(**self.args.__dict__)
        self.solar_heating = solar(**self.args.__dict__)
        self.convective_cooling = convective(**self.args.__dict__)
        self.radiative_cooling = radiative(**self.args.__dict__)
        self.precipitation_cooling = precipitation(**self.args.__dict__)
        self.args.compress()

    def update(self) -> None:
        self.args.extend_to_max_len()
        self.joule_heating.__init__(**self.args.__dict__)
        self.solar_heating.__init__(**self.args.__dict__)
        self.convective_cooling.__init__(**self.args.__dict__)
        self.radiative_cooling.__init__(**self.args.__dict__)
        self.precipitation_cooling.__init__(**self.args.__dict__)
        self.args.compress()

    def balance(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        return (
            self.joule_heating.value(conductor_temperature)
            + self.solar_heating.value(conductor_temperature)
            - self.convective_cooling.value(conductor_temperature)
            - self.radiative_cooling.value(conductor_temperature)
            - self.precipitation_cooling.value(conductor_temperature)
        )

    @staticmethod
    def powers() -> tuple[PowerType, PowerType, PowerType, PowerType, PowerType]:
        return (
            PowerType.JOULE,
            PowerType.SOLAR,
            PowerType.CONVECTION,
            PowerType.RADIATION,
            PowerType.RAIN,
        )

    @abstractmethod
    def steady_temperature(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def transient_temperature(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def steady_intensity(self) -> pd.DataFrame:
        raise NotImplementedError


def reshape(input_array: numberArrayLike, nb_row: int, nb_columns: int) -> numberArray:
    """
    Reshape the input array to the specified dimensions (nr, nc) if possible.

    Args:
        input_array (numberArrayLike): Input array to be reshaped.
        nb_row (int): Desired number of rows for the reshaped array.
        nb_columns (int): Desired number of columns for the reshaped array.

    Returns:
        numberArray: Reshaped array of size (nb_row, nb_columns). If reshaping is not possible,
            returns an array filled with the input_value repeated to fill the dimension (nb_row, nb_columns).

    Raises:
        AttributeError: If the input_array has an invalid shape that cannot be reshaped.
    """
    reshaped_array = ndarray
    try:
        input_shape = input_array.shape
        if len(input_shape) == 1:
            if nb_row == input_shape[0]:
                reshaped_array = np.column_stack(nb_columns * (input_array,))
            elif nb_columns == input_shape[0]:
                reshaped_array = np.vstack(nb_row * (input_array,))
        elif len(input_shape) == 0:
            raise AttributeError()
        else:
            reshaped_array = np.reshape(input_array, (nb_row, nb_columns))
    except AttributeError:
        reshaped_array = input_array * np.ones(
            (nb_row, nb_columns), dtype=type(input_array)
        )
    return reshaped_array


def _set_dates(
    month: floatArrayLike,
    day: floatArrayLike,
    hour: floatArrayLike,
    time: floatArray,
    input_size: int,
) -> Tuple[intArray, intArray, floatArray]:
    """
    Set months, days and hours as 2D arrays.

    This function is used in transient temperature computations. Inputs month,
    day and hour are floats or 1D arrays of size input_size; input t is a time vector of
    size time_size with evaluation times in seconds. It sets arrays months, days and
    hours, of size (time_size, input_size) such that
        months[i, j] = datetime(month[j], day[j], hour[j]) + t[i] .

    Args:
        month (floatArrayLike): Array of floats or float representing the months.
        day (floatArrayLike): Array of floats or float representing the days.
        hour (floatArrayLike): Array of floats or float representing the hours.
        time (floatArray): Array of floats representing the time vector in seconds.
        input_size (int): Size of the input arrays month, day, and hour.

    Returns:
    Tuple[intArray, intArray, floatArray]:
        - months (intArray): 2D array of shape (time_size, input_size) with month values.
        - days (intArray): 2D array of shape (time_size, input_size) with day values.
        - hours (floatArray): 2D array of shape (time_size, input_size) with hour values.
    """
    ones_int = np.ones((input_size,), dtype=int)
    ones_float = np.ones((input_size,), dtype=float)
    month2 = month * ones_int
    day2 = day * ones_int
    hour2 = hour * ones_float

    time_size = len(time)
    months = np.zeros((time_size, input_size), dtype=int)
    days = np.zeros((time_size, input_size), dtype=int)
    hours = np.zeros((time_size, input_size), dtype=float)

    time_delta = np.array(
        [datetime.timedelta()]
        + [
            datetime.timedelta(seconds=float(time[i] - time[i - 1]))
            for i in range(1, time_size)
        ]
    )

    for j in range(input_size):
        hour_j = int(np.floor(hour2[j]))
        time_delta_j = datetime.timedelta(seconds=float(3600.0 * (hour2[j] - hour_j)))
        t0 = (
            datetime.datetime(year=2000, month=month2[j], day=day2[j], hour=hour_j)
            + time_delta_j
        )
        ts = pd.Series(t0 + time_delta)
        months[:, j] = ts.dt.month
        days[:, j] = ts.dt.day
        hours[:, j] = (
            ts.dt.hour
            + ts.dt.minute / 60.0
            + (ts.dt.second + ts.dt.microsecond * 1.0e-06) / 3600.0
        )

    return months, days, hours
