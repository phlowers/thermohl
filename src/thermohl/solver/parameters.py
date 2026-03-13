# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timezone
from typing import Optional, Any, KeysView, Iterable
import numpy as np


class DEFAULT_PARAMETERS:
    tmin = -99.0
    tmax = +999.0
    tol = 1.0e-09
    maxiter = 64
    imin = 0.0
    imax = 9999.0


class Parameters:
    """Object to store Solver args in a dict-like manner."""

    def __init__(self, input_dict: Optional[dict[str, Any]] = None):
        # add default values
        self._set_default_values()
        # use values from input dict
        if input_dict is None:
            return
        arg_keys = self.keys()
        for input_key, input_value in input_dict.items():
            if input_key in arg_keys and input_value is not None:
                self[input_key] = input_value

    def _set_default_values(self) -> None:
        """Set default values."""
        # position
        self.latitude = 45.0  # latitude (deg)
        self.longitude = 0.0  # longitude (deg)
        self.altitude = 0.0  # altitude (m)
        self.cable_azimuth = 0.0  # cable_azimuth (deg)
        self.datetime_utc = datetime(2000, 3, 21, 12, tzinfo=timezone.utc)

        # weather and mesurement
        self.measured_global_radiation = np.nan  # solar irradiance
        self.ambient_temperature = 15.0  # ambient temperature (C)
        self.ambient_pressure = 1.0e05  # ambient pressure (Pa)
        self.relative_humidity = 0.8  # relative humidity (none, in [0, 1])
        self.precipitation_rate = 0.0  # rain precipitation rate (m.s**-1)
        self.wind_speed = 0.0  # wind speed (m.s**-1)
        self.wind_azimuth = 90.0  # wind_azimuth (deg, regarding north)
        self.nebulosity = 0  # nebulosity (1)
        self.albedo = 0.15  # albedo (1)
        # coefficient for air pollution from 0 (clean) to 1 (polluted)
        self.turbidity = 0.1
        self.transit = 100.0  # transit intensity (A)

        # conductor
        self.linear_mass = 1.5  # mass per unit length (kg.m**-1)
        self.core_diameter = 1.9e-02  # core diameter (m)
        self.outer_diameter = 3.0e-02  # external (global) diameter (m)
        self.core_area = 2.84e-04  # core section (m**2)
        self.outer_area = 7.07e-04  # external (global) section (m**2)
        self.roughness_ratio = 4.0e-02  # roughness (1)
        # radial thermal conductivity (W.m**-1.K**-1)
        self.radial_thermal_conductivity = 1.0
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
        # temperature for linear_resistance_temp_high measurement (°C)
        self.temp_high = 60.0
        # temperature for linear_resistance_temp_low measurement (°C)
        self.temp_low = 20.0

    def keys(self) -> KeysView[str]:
        """Get list of members as dict keys."""
        return self.__dict__.keys()

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def get_number_of_computations(self) -> int:
        return max(
            (len(self[k]) for k in self.keys() if isinstance(self[k], Iterable)),
            default=1,
        )

    def extend(self) -> None:
        """
        Extend all compressed elements in the Args dictionary to the right length, ie the number of computations.
        If the element is a list, it already has the right length.
        If the element is a scalar, it is replaced with a list of the right length filled with the scalar value.
        """
        number_of_computations = self.get_number_of_computations()
        for key in self.keys():
            if not isinstance(self[key], Iterable):
                self[key] = np.array(number_of_computations * [self[key]])

    def compress(self) -> None:
        """
        Compress the elements in the Args dictionary by replacing lists containing a unique value
        with this value.
        """
        for key in self.keys():
            u = np.unique(self[key])
            if len(u) == 1:
                self[key] = u[0]
