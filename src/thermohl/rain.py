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
        ambient_temp_c,
        wind_speed,
        wind_angle,
        outer_diameter,
        air_pressure_pa,
        relative_humidity,
        rain_rate_ms,
        **kwargs,
    ):
        # maksic = (
        #     PrecipitationCooling._evap(
        #         conductor_temperature,
        #         altitude,
        #         ambient_temp_c,
        #         wind_speed,
        #         wind_angle,
        #         outer_diameter,
        #         air_pressure_pa,
        #         relative_humidity,
        #         rain_rate_ms,
        #     )
        #     + PrecipitationCooling._imp(
        #         conductor_temperature,
        #         ambient_temp_c,
        #         wind_speed,
        #         outer_diameter,
        #         rain_rate_ms,
        #     )
        # )
        pytlak = PrecipitationCooling._evap(
            conductor_temperature,
            altitude,
            ambient_temp_c,
            wind_speed,
            wind_angle,
            outer_diameter,
            air_pressure_pa,
            relative_humidity,
            rain_rate_ms,
        )
        return pytlak

    @staticmethod
    def _ma(
        ambient_temp_c,
        wind_speed,
        outer_diameter,
        rain_rate_ms,
        snow_rate_ms,
    ):
        # ! -- precipitation_rate and ps in m.s**-1
        water_density = thermodynamics.Water.volumic_mass(ambient_temp_c)
        rain_mass_flux = np.sqrt(
            (rain_rate_ms * water_density) ** 2
            + (wind_speed * 23.589 * rain_rate_ms**0.8460) ** 2
        )
        snow_mass_flux = np.sqrt(
            (snow_rate_ms * water_density) ** 2
            + (wind_speed * 142.88 * snow_rate_ms**0.9165) ** 2
        )
        return outer_diameter * (rain_mass_flux + snow_mass_flux)

    @staticmethod
    def _me(
        conductor_temperature,
        altitude,
        ambient_temp_c,
        wind_speed,
        wind_angle,
        outer_diameter,
        air_pressure_pa,
        relative_humidity,
    ):
        film_temperature = 0.5 * (conductor_temperature + ambient_temp_c)
        temperature_delta = conductor_temperature - ambient_temp_c
        air_density = air.IEEE.volumic_mass(film_temperature, altitude)
        convective_cooling = ieee.ConvectiveCooling._value_forced(
            film_temperature,
            temperature_delta,
            air_density,
            wind_speed,
            outer_diameter,
            wind_angle,
        )
        heat_transfer_coeff = convective_cooling / (
            np.pi * outer_diameter * temperature_delta
        )
        mass_ratio = 18.015 / 28.9647
        wetted_perimeter_m = np.pi * outer_diameter
        air_heat_capacity = thermodynamics.Air.heat_capacity(
            temp_k=olla.air.kelvin(film_temperature)
        )
        vapor_pressure_conductor = thermodynamics.Water.vapor_pressure(
            temp_k=olla.air.kelvin(conductor_temperature)
        )
        vapor_pressure_ambient = thermodynamics.Water.vapor_pressure(
            temp_k=olla.air.kelvin(ambient_temp_c)
        )
        evaporation_mass_flux = (
            wetted_perimeter_m
            * heat_transfer_coeff
            * mass_ratio
            * (vapor_pressure_conductor - relative_humidity * vapor_pressure_ambient)
            / (air_heat_capacity * air_pressure_pa)
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
        ambient_temp_c,
        wind_speed,
        wind_angle,
        outer_diameter,
        air_pressure_pa,
        relative_humidity,
        rain_rate_ms,
        snow_rate_ms,
    ):
        return np.minimum(
            PrecipitationCooling._ma(
                ambient_temp_c,
                wind_speed,
                outer_diameter,
                rain_rate_ms,
                snow_rate_ms,
            ),
            PrecipitationCooling._me(
                conductor_temperature,
                altitude,
                ambient_temp_c,
                wind_speed,
                wind_angle,
                outer_diameter,
                air_pressure_pa,
                relative_humidity,
            ),
        )

    @staticmethod
    def _evap(
        conductor_temperature,
        altitude,
        ambient_temp_c,
        wind_speed,
        wind_angle,
        outer_diameter,
        air_pressure_pa,
        relative_humidity,
        rain_rate_ms,
        snow_rate_ms=0.0,
    ):
        mass_flux = PrecipitationCooling._mass_flux(
            conductor_temperature,
            altitude,
            ambient_temp_c,
            wind_speed,
            wind_angle,
            outer_diameter,
            air_pressure_pa,
            relative_humidity,
            rain_rate_ms,
            snow_rate_ms,
        )
        latent_heat_vap = thermodynamics.Water.heat_of_vaporization()
        water_heat_capacity = thermodynamics.Water.heat_capacity(
            temp_k=olla.air.kelvin(conductor_temperature)
        )
        boiling_temp_k = thermodynamics.Water.boiling_point(pressure_pa=air_pressure_pa)
        boiling_temp_c = air.celsius(boiling_temp_k)
        effective_temp_c = np.minimum(conductor_temperature, boiling_temp_c)
        latent_heat_fusion = thermodynamics.Ice.heat_of_fusion()
        ice_heat_capacity = thermodynamics.Ice.heat_capacity()
        rain_cooling = mass_flux * (
            latent_heat_vap + water_heat_capacity * (effective_temp_c - ambient_temp_c)
        )
        snow_cooling = mass_flux * (
            latent_heat_vap
            + water_heat_capacity * conductor_temperature
            + latent_heat_fusion * ice_heat_capacity * ambient_temp_c
        )
        return np.maximum(
            rain_cooling + 0.0 * snow_cooling, np.zeros_like(conductor_temperature)
        )

    @staticmethod
    def _imp(
        conductor_temperature,
        ambient_temp_c,
        wind_speed,
        outer_diameter,
        rain_rate_ms,
        snow_rate_ms=0.0,
    ):
        water_heat_capacity = thermodynamics.Water.heat_capacity(
            temp_k=olla.air.kelvin(conductor_temperature)
        )
        return (
            0.71
            * water_heat_capacity
            * (conductor_temperature - ambient_temp_c)
            * PrecipitationCooling._ma(
                ambient_temp_c,
                wind_speed,
                outer_diameter,
                rain_rate_ms,
                snow_rate_ms,
            )
        )
