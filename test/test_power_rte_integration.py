# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd

from thermohl.power import rte
from thermohl.solver import Args
from thermohl.solver.enums.variable_type import VariableType


class ExcelSheet:
    """Object to compare power terms from rte's excel sheet version 7."""

    def __init__(self, dic):
        self.args = Args(dic)
        self.args.nbc = dic["nbc"]

    def joule_heating(self, ambient_temperature, transit=None):
        if transit is None:
            transit = self.args[VariableType.TRANSIT.value]
        core_diameter = self.args["core_diameter"]
        outer_diameter = self.args["outer_diameter"]
        Rdc = self.args["linear_resistance_dc_20c"] * (
            1.0
            + self.args["temperature_coeff_linear"] * (ambient_temperature - 20.0)
            + self.args["temperature_coeff_quadratic"]
            * (ambient_temperature - 20.0) ** 2
        )
        z = (
            8
            * np.pi
            * 50.0
            * (outer_diameter - core_diameter) ** 2
            / ((outer_diameter**2 - core_diameter**2) * 1.0e07 * Rdc)
        )
        a = 7 * z**2 / (315 + 3 * z**2)
        b = 56 / (211 + z**2)
        beta = 1.0 - core_diameter / outer_diameter
        kep = 1 + a * (1.0 - 0.5 * beta - b * beta**2)
        kem = np.where(
            (core_diameter > 0.0) & (self.args["nbc"] == 3),
            self.args["magnetic_coeff"]
            + self.args["magnetic_coeff_per_a"]
            * transit
            / (self.args["outer_area"] - self.args["core_area"])
            * 1.0e-06,
            1.0,
        )
        Rac = Rdc * kep * kem
        return Rac * self.args[VariableType.TRANSIT.value] ** 2

    def solar_heating(self):
        csm = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        O4 = csm[self.args["month"] - 1] + self.args["day"]
        O5 = self.args["hour"]
        O6 = np.deg2rad(self.args["latitude"])
        Q4 = np.deg2rad(23.46 * np.sin(np.deg2rad((284 + O4) / 365 * 360)))
        Q5 = np.deg2rad((O5 - 12) * 15)
        Q6 = np.rad2deg(
            np.arcsin(np.cos(O6) * np.cos(Q4) * np.cos(Q5) + np.sin(O6) * np.sin(Q4))
        )
        q = np.maximum(
            -42
            + 63.8 * Q6
            - 1.922 * Q6**2
            + 0.03469 * Q6**3
            - 0.000361 * Q6**4
            + 0.000001943 * Q6**5
            - 0.00000000408 * Q6**6,
            0.0,
        )
        Q7 = np.sin(Q5) / (np.sin(O6) * np.cos(Q5) - np.cos(O6) * np.tan(Q4))
        Q8 = np.deg2rad(180 + np.rad2deg(np.arctan(Q7)))
        O7 = np.pi / 2
        O2 = np.arccos(np.cos(np.deg2rad(Q6)) * np.cos(Q8 - O7))
        q *= np.sin(O2)
        return q * self.args["outer_diameter"] * self.args["solar_absorptivity"]

    def convective_cooling(self, Ts):
        outer_diameter = self.args["outer_diameter"]
        Tf = 0.5 * (Ts + self.args["ambient_temperature"])
        lm = 0.02424 + 0.00007477 * Tf - 0.000000004407 * Tf**2
        air_density = (
            1.293
            - 0.0001525 * self.args["altitude"]
            + 0.00000000638 * self.args["altitude"] ** 2
        ) / (1 + 0.00367 * Tf)
        dynamic_viscosity = (0.000001458 * (Tf + 273) ** 1.5) / (Tf + 383.4)
        Re = (
            self.args["wind_speed"]
            * self.args["outer_diameter"]
            * air_density
            / dynamic_viscosity
        )
        F = np.maximum(1.01 + 1.35 * Re**0.52, 0.754 * Re**0.6)
        wind_angle = np.deg2rad(self.args["wind_angle"])
        K = (
            1.194
            - np.cos(wind_angle)
            + 0.194 * np.cos(2 * wind_angle)
            + 0.368 * np.sin(2 * wind_angle)
        )
        PCn = (
            3.645
            * air_density**0.5
            * outer_diameter**0.75
            * np.sign(Ts - self.args["ambient_temperature"])
            * np.abs(Ts - self.args["ambient_temperature"]) ** 1.25
        )
        PCf = F * lm * K * (Ts - self.args["ambient_temperature"])
        # print(f"re={Re}, kp={K}, lam={lm}")
        # print(f"pcn={PCn}, pcf={PCf}")
        return np.maximum(PCn, PCf)

    def radiative_cooling(self, Ts):
        outer_diameter = self.args["outer_diameter"]
        return (
            17.8
            * outer_diameter
            * self.args["emissivity"]
            * (
                ((273 + Ts) / 100) ** 4
                - ((273 + self.args["ambient_temperature"]) / 100) ** 4
            )
        )


def excel_conductor_data():
    """Get conductor data from excel sheet (hard-coded)."""
    df = pd.DataFrame(
        dict(
            conductor=[
                "ACSS1317",
                "Aster228",
                "Aster570",
                "Crocus412",
                "Pastel228",
                "Petunia612",
            ],
            outer_diameter=[44.0, 19.6, 31.06, 26.4, 19.6, 32.1],
            core_diameter=[21.28, 0.0, 0.0, 12.0, 8.4, 13.25],
            outer_area=[1317, 228, 570, 412, 228, 612],
            core_area=[0, 0, 0, 0, 0, 0],
            B=[1049, 228, 570, 323, 185, 508],
            linear_resistance_dc_20c=[0.0272, 0.146, 0.0583, 0.089, 0.18, 0.0657],
            temperature_coeff_linear=[0.004, 0.0036, 0.0036, 0.004, 0.0036, 0.0036],
            magnetic_coeff=[1.006, 1.0, 1.0, 1.0, 1.0, 1.006],
            magnetic_coeff_per_a=[0.016, 0.0, 0.0, 0.0, 0.0, 0.016],
            temperature_coeff_quadratic=[
                8.0e-07,
                8.0e-07,
                8.0e-07,
                8.0e-07,
                8.0e-07,
                8.0e-07,
            ],
            nbc=[3, 0, 0, 2, 2, 3],
        )
    )

    df["core_area"] = df["outer_area"] - df["B"]
    df.drop(columns=["B"], inplace=True)
    df["outer_diameter"] *= 1.0e-03
    df["core_diameter"] *= 1.0e-03
    df["outer_area"] *= 1.0e-06
    df["core_area"] *= 1.0e-06
    df["linear_resistance_dc_20c"] *= 1.0e-03

    return df


def scenarios():
    """Get list of hard-coded scenarios to test."""
    dic = dict(
        conductor=[
            "Aster228",
            "Pastel228",
            "Petunia612",
            "Petunia612",
            "ACSS1317",
            "ACSS1317",
            "Aster228",
            "Aster228",
            "Aster228",
            "Aster228",
        ],
        ambient_temperature=[
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
        ],
        wind_speed=[3.0, 3.0, 3.0, 0.0, 0.0, 3.0, 0.6, 0.6, 0.6, 0.6],
        wind_angle=[90.0, 90.0, 90.0, 45.0, 45.0, 90.0, 90.0, 90.0, 90.0, 90.0],
        measured_solar_irradiance=[
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        latitude=[46.0, 46.0, 46.0, 46.0, 46.0, 46.0, 46.0, 46.0, 46.0, 46.0],
        altitude=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        azimuth=90.0,
        transit=[
            1000.0,
            1000.0,
            1800.0,
            1100.0,
            3000.0,
            4000.0,
            700.0,
            700.0,
            700.0,
            700.0,
        ],
        solar_absorptivity=0.9,
        emissivity=0.8,
        turbidity=0.0,
        month=[3, 3, 3, 3, 3, 3, 3, 6, 6, 6],
        day=[7, 7, 7, 7, 7, 7, 7, 21, 21, 21],
        hour=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 12.0, 19.0, 12.0],
    )
    df = pd.DataFrame(dic)
    dg = excel_conductor_data()
    df = pd.merge(df, dg, on="conductor", how="left").drop(columns="conductor")
    return df


def test_compare_power():
    """Compare computed values to hard-coded ones from ieee guide [find ref]."""

    conductor_temperature = np.linspace(-50, +250, 999)

    ds = scenarios()
    n = len(ds)
    ds = pd.concat(len(conductor_temperature) * (ds,)).reset_index(drop=True)
    conductor_temperature = np.concatenate([n * (t,) for t in conductor_temperature])

    from thermohl.utils import df2dct

    d1 = df2dct(ds)
    ds["wind_angle"] = np.rad2deg(
        np.arcsin(np.sin(np.deg2rad(np.abs(ds["azimuth"] - ds["wind_angle"]) % 180.0)))
    )
    d2 = df2dct(ds)
    del (ds, n)

    pj = rte.JouleHeating(**d1)
    ps = rte.SolarHeating(**d1)
    precipitation_cooling = rte.ConvectiveCooling(**d1)
    precipitation_rate = rte.RadiativeCooling(**d1)
    ex = ExcelSheet(d2)

    assert np.allclose(
        ex.joule_heating(conductor_temperature), pj.value(conductor_temperature)
    )
    assert np.allclose(ex.solar_heating(), ps.value(0.0))
    assert np.allclose(
        ex.convective_cooling(conductor_temperature),
        precipitation_cooling.value(conductor_temperature),
    )
    assert np.allclose(
        ex.radiative_cooling(conductor_temperature),
        precipitation_rate.value(conductor_temperature),
    )


def test_solar_heating():
    # adapted from devin repo dev/test/test_rte.py

    n = 5
    ones = np.ones(n)

    latitude = np.array([40.0, 46.0, 46.0, 46.0, 46.0])
    azimuth = np.array([90.0, 0.0, 0.0, 0.0, 0.0])
    month = np.array([7, 3, 3, 3, 3])
    day = np.array([19, 7, 14, 7, 7])
    hour = np.array([14.0, 12.0, 17.0, 12.0, 12.0])
    outer_diameter = 4.4e-02 * ones
    solar_absorptivity = 0.9 * ones

    p = np.array([34.9, 21.9357, 13.95, 21.9357, 21.9357])
    s = rte.SolarHeating(
        latitude,
        azimuth,
        month,
        day,
        hour,
        outer_diameter,
        solar_absorptivity,
        measured_solar_irradiance=np.nan,
    )

    assert np.allclose(p, s.value(ones), 0.1)
