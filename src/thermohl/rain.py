# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

# mypy: ignore-errors
"""Rain cooling power term.

Based on *Modelling precipitation cooling of overhead conductors*, Pytlak et
al., 2011 (https://doi.org/10.1016/j.epsr.2011.06.004) and *Dynamic thermal
rating of power lines in raining conditions-model and measurements*, Maksic et
al., 2016 (https://doi.org/10.1109/PESGM.2016.7741611).
"""

import numpy as np

from thermohl import air
from thermohl import thermodynamics
from thermohl.power import olla, ieee
from thermohl.utils import PowerTerm


class PrecipitationCooling(PowerTerm):
    """Precipitation cooling term."""

    def value(
        self,
        conductor_temperature,
        altitude,
        ambient_temp,
        wind_speed,
        wind_azimuth,
        outer_diameter,
        air_pressure,
        relative_humidity,
        rain_rate,
        **kwargs,
    ):
        # maksic = (
        #     PrecipitationCooling._evap(
        #         conductor_temperature,
        #         altitude,
        #         ambient_temp,
        #         wind_speed,
        #         wind_azimuth,
        #         outer_diameter,
        #         air_pressure,
        #         relative_humidity,
        #         rain_rate,
        #     )
        #     + PrecipitationCooling._imp(
        #         conductor_temperature,
        #         ambient_temp,
        #         wind_speed,
        #         outer_diameter,
        #         rain_rate,
        #     )
        # )
        pytlak = PrecipitationCooling._evap(
            conductor_temperature,
            altitude,
            ambient_temp,
            wind_speed,
            wind_azimuth,
            outer_diameter,
            air_pressure,
            relative_humidity,
            rain_rate,
        )
        return pytlak

    @staticmethod
    def _ma(
        ambient_temp,
        wind_speed,
        outer_diameter,
        rain_rate,
        snow_rate,
    ):
        # ! -- precipitation_rate and ps in m.s**-1
        water_density = thermodynamics.Water.volumic_mass(ambient_temp)
        rain_mass_flux = np.sqrt(
            (rain_rate * water_density) ** 2
            + (wind_speed * 23.589 * rain_rate**0.8460) ** 2
        )
        snow_mass_flux = np.sqrt(
            (snow_rate * water_density) ** 2
            + (wind_speed * 142.88 * snow_rate**0.9165) ** 2
        )
        return outer_diameter * (rain_mass_flux + snow_mass_flux)

    @staticmethod
    def _me(
        conductor_temperature,
        altitude,
        ambient_temp,
        wind_speed,
        wind_azimuth,
        outer_diameter,
        air_pressure,
        relative_humidity,
    ):
        film_temperature = 0.5 * (conductor_temperature + ambient_temp)
        temperature_delta = conductor_temperature - ambient_temp
        air_density = air.IEEE.volumic_mass(film_temperature, altitude)
        convective_cooling = ieee.ConvectiveCooling._value_forced(
            film_temperature,
            temperature_delta,
            air_density,
            wind_speed,
            outer_diameter,
            wind_azimuth,
        )
        heat_transfer_coeff = convective_cooling / (
            np.pi * outer_diameter * temperature_delta
        )
        mass_ratio = 18.015 / 28.9647
        wetted_perimeter = np.pi * outer_diameter
        air_heat_capacity = thermodynamics.Air.heat_capacity(
            temperature=olla.air.kelvin(film_temperature)
        )
        vapor_pressure_conductor = thermodynamics.Water.vapor_pressure(
            temperature=olla.air.kelvin(conductor_temperature)
        )
        vapor_pressure_ambient = thermodynamics.Water.vapor_pressure(
            temperature=olla.air.kelvin(ambient_temp)
        )
        evaporation_mass_flux = (
            wetted_perimeter
            * heat_transfer_coeff
            * mass_ratio
            * (vapor_pressure_conductor - relative_humidity * vapor_pressure_ambient)
            / (air_heat_capacity * air_pressure)
        )
        return np.where(
            ~np.isclose(temperature_delta, 0.0, atol=0.0005),
            evaporation_mass_flux,
            np.zeros_like(conductor_temperature),
        )

    @staticmethod
    def _mass_flux(
        conductor_temperature,
        altitude,
        ambient_temp,
        wind_speed,
        wind_azimuth,
        outer_diameter,
        air_pressure,
        relative_humidity,
        rain_rate,
        snow_rate,
    ):
        return np.minimum(
            PrecipitationCooling._ma(
                ambient_temp,
                wind_speed,
                outer_diameter,
                rain_rate,
                snow_rate,
            ),
            PrecipitationCooling._me(
                conductor_temperature,
                altitude,
                ambient_temp,
                wind_speed,
                wind_azimuth,
                outer_diameter,
                air_pressure,
                relative_humidity,
            ),
        )

    @staticmethod
    def _evap(
        conductor_temperature,
        altitude,
        ambient_temp,
        wind_speed,
        wind_azimuth,
        outer_diameter,
        air_pressure,
        relative_humidity,
        rain_rate,
        snow_rate=0.0,
    ):
        mass_flux = PrecipitationCooling._mass_flux(
            conductor_temperature,
            altitude,
            ambient_temp,
            wind_speed,
            wind_azimuth,
            outer_diameter,
            air_pressure,
            relative_humidity,
            rain_rate,
            snow_rate,
        )
        latent_heat_vap = thermodynamics.Water.heat_of_vaporization()
        water_heat_capacity = thermodynamics.Water.heat_capacity(
            temperature=olla.air.kelvin(conductor_temperature)
        )
        boiling_temp = thermodynamics.Water.boiling_point(pressure=air_pressure)
        boiling_temp_c = air.celsius(boiling_temp)
        effective_temp = np.minimum(conductor_temperature, boiling_temp_c)
        latent_heat_fusion = thermodynamics.Ice.heat_of_fusion()
        ice_heat_capacity = thermodynamics.Ice.heat_capacity()
        rain_cooling = mass_flux * (
            latent_heat_vap + water_heat_capacity * (effective_temp - ambient_temp)
        )
        snow_cooling = mass_flux * (
            latent_heat_vap
            + water_heat_capacity * conductor_temperature
            + latent_heat_fusion * ice_heat_capacity * ambient_temp
        )
        return np.maximum(
            rain_cooling + 0.0 * snow_cooling, np.zeros_like(conductor_temperature)
        )

    @staticmethod
    def _imp(
        conductor_temperature,
        ambient_temp,
        wind_speed,
        outer_diameter,
        rain_rate,
        snow_rate=0.0,
    ):
        water_heat_capacity = thermodynamics.Water.heat_capacity(
            temperature=olla.air.kelvin(conductor_temperature)
        )
        return (
            0.71
            * water_heat_capacity
            * (conductor_temperature - ambient_temp)
            * PrecipitationCooling._ma(
                ambient_temp,
                wind_speed,
                outer_diameter,
                rain_rate,
                snow_rate,
            )
        )
