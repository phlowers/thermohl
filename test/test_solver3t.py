# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl import solver
from thermohl.solver import HeatEquationType
from thermohl.solver.entities import (
    CableType,
    ModelType,
    VariableType,
    TemperatureType,
    TargetType,
)

_nprs = 123456


def _solvers(dic=None):
    return [
        solver._factory(dic=dic, heat_equation=heat_equation, model=m)
        for heat_equation in [
            HeatEquationType.THREE_TEMPERATURES,
            HeatEquationType.THREE_TEMPERATURES_LEGACY,
        ]
        for m in [
            ModelType.RTE,
            ModelType.CIGRE,
            ModelType.IEEE,
            ModelType.OLLA,
        ]
    ]


def test_balance():
    # NB : this one fails only with 'cigre' powers

    tol = 1.0e-09

    rng = np.random.default_rng(_nprs)

    N = 9999
    dic = {
        "latitude": rng.random(N) * 9 + 42.0,
        "altitude": rng.random(N) * 1600.0,
        "cable_azimuth": rng.random(N) * 360.0,
        "month": rng.integers(1, 13, N),
        "day": rng.integers(1, 31, N),
        "hour": rng.integers(0, 24, N),
        "ambient_temperature": rng.random(N) * 30.0,
        "wind_speed": rng.random(N) * 7.0,
        "wind_azimuth": rng.random(N) * 90.0,
        "transit": rng.random(N) * 3960 + 40.0,
        "core_diameter": rng.integers(2, size=N)
        * solver.default_values()["core_diameter"],
    }

    for s in _solvers(dic):
        # compute guess with 1t solver
        s1 = solver._factory(
            dic=dic,
            heat_equation=HeatEquationType.ONE_TEMPERATURE,
            model=ModelType.IEEE,
        )
        steady_temperature_1t = s1.steady_temperature(
            tol=2.0, maxiter=16, return_err=False, return_power=False
        )
        steady_temperature_1t = steady_temperature_1t[VariableType.TEMPERATURE.value]
        # 3t solve
        steady_temperature_3t = s.steady_temperature(
            surface_temperature_guess=steady_temperature_1t,
            core_temperature_guess=steady_temperature_1t,
            return_err=True,
            return_power=True,
            tol=tol,
            maxiter=64,
        )
        # checks
        assert np.all(steady_temperature_3t[VariableType.ERROR.value] < tol)
        assert np.allclose(
            s.balance(
                surface_temperature=steady_temperature_3t[
                    TemperatureType.SURFACE.value
                ],
                core_temperature=steady_temperature_3t[TemperatureType.CORE.value],
            ),
            0.0,
            atol=tol,
        )
        assert np.allclose(
            s.morgan(
                surface_temperature=steady_temperature_3t[
                    TemperatureType.SURFACE.value
                ],
                core_temperature=steady_temperature_3t[TemperatureType.CORE.value],
            ),
            0.0,
            atol=tol,
        )


def test_consistency():
    tol = 1.0e-09

    rng = np.random.default_rng(_nprs)

    N = 9999

    dic = {
        "latitude": rng.random(N) * 9 + 42.0,
        "altitude": rng.random(N) * 1600.0,
        "cable_azimuth": rng.random(N) * 360.0,
        "month": rng.integers(1, 13, N),
        "day": rng.integers(1, 31, N),
        "hour": rng.integers(0, 24, N),
        "ambient_temperature": rng.random(N) * 30.0,
        "wind_speed": rng.random(N) * 7.0,
        "wind_azimuth": rng.random(N) * 90.0,
        "core_diameter": rng.integers(2, size=N)
        * solver.default_values()["core_diameter"],
    }

    for s in _solvers(dic):
        d = {
            TargetType.SURFACE: TemperatureType.SURFACE,
            TargetType.AVERAGE: TemperatureType.AVERAGE,
            TargetType.CORE: TemperatureType.CORE,
        }

        for location, temperature_at_location in d.items():
            # solve intensity with different targets
            steady_temperature_1 = s.steady_intensity(
                max_conductor_temperature=100.0,
                target=location,
                return_err=True,
                return_power=True,
                tol=1.0e-09,
                maxiter=64,
            )
            assert np.all(steady_temperature_1[VariableType.ERROR.value] < tol)
            # set args intensity to newly founds ampacities
            s.args.I = steady_temperature_1[VariableType.TRANSIT.value]
            s.update()
            assert np.allclose(
                s.balance(
                    surface_temperature=steady_temperature_1[
                        TemperatureType.SURFACE.value
                    ],
                    core_temperature=steady_temperature_1[TemperatureType.CORE.value],
                ),
                0.0,
                atol=tol,
            )
            assert np.allclose(
                s.morgan(
                    surface_temperature=steady_temperature_1[
                        TemperatureType.SURFACE.value
                    ],
                    core_temperature=steady_temperature_1[TemperatureType.CORE.value],
                ),
                0.0,
                atol=tol,
            )
            # 3t solve
            steady_temperature_2 = s.steady_temperature(
                surface_temperature_guess=steady_temperature_1[
                    TemperatureType.SURFACE.value
                ].round(1),
                core_temperature_guess=steady_temperature_1[
                    TemperatureType.CORE.value
                ].round(1),
                return_err=True,
                return_power=True,
                tol=1.0e-09,
                maxiter=64,
            )
            # check consistency
            assert np.allclose(
                steady_temperature_2[temperature_at_location.value], 100.0
            )


def test_steady_intensity_cable_type_scalar_homogeneous():
    for s in _solvers():
        result_with_specified_cable_type = s.steady_intensity(
            max_conductor_temperature=100.0,
            cable_type=CableType.HOMOGENEOUS,
            return_err=True,
            return_power=True,
        )

        result_with_specified_target = s.steady_intensity(
            max_conductor_temperature=100.0,
            target=TargetType.AVERAGE,
            return_err=True,
            return_power=True,
        )

        assert np.allclose(
            result_with_specified_cable_type[VariableType.TRANSIT.value],
            result_with_specified_target[VariableType.TRANSIT.value],
        )


def test_steady_intensity_cable_type_scalar_bimetallic():
    for s in _solvers():
        result_with_specified_cable_type = s.steady_intensity(
            max_conductor_temperature=100.0,
            cable_type=CableType.BIMETALLIC,
            return_err=True,
            return_power=True,
        )

        result_with_specified_target = s.steady_intensity(
            max_conductor_temperature=100.0,
            target=TargetType.CORE,
            return_err=True,
            return_power=True,
        )

        assert np.allclose(
            result_with_specified_cable_type[VariableType.TRANSIT.value],
            result_with_specified_target[VariableType.TRANSIT.value],
        )


def test_steady_intensity_cable_type_list():
    rng = np.random.default_rng(_nprs)

    N = 3

    dic = {
        "latitude": rng.random(N) * 9 + 42.0,
        "altitude": rng.random(N) * 1600.0,
        "azimuth": rng.random(N) * 360.0,
        "month": rng.integers(1, 13, N),
        "day": rng.integers(1, 31, N),
        "hour": rng.integers(0, 24, N),
        "ambient_temperature": rng.random(N) * 30.0,
        "wind_speed": rng.random(N) * 7.0,
        "wind_angle": rng.random(N) * 90.0,
        "core_diameter": rng.integers(2, size=N)
        * solver.default_values()["core_diameter"],
    }

    cable_type = np.array(
        [CableType.BIMETALLIC, CableType.HOMOGENEOUS, CableType.HOMOGENEOUS]
    )
    target = np.array([TargetType.CORE, TargetType.AVERAGE, TargetType.AVERAGE])

    for s in _solvers(dic):
        result_with_specified_cable_type = s.steady_intensity(
            max_conductor_temperature=100.0,
            cable_type=cable_type,
            return_err=True,
            return_power=True,
        )

        result_with_specified_target = s.steady_intensity(
            max_conductor_temperature=100.0,
            target=target,
            return_err=True,
            return_power=True,
        )

        assert np.allclose(
            result_with_specified_cable_type[VariableType.TRANSIT.value],
            result_with_specified_target[VariableType.TRANSIT.value],
        )


def test_steady_intensity_cable_type_and_target():
    # Both cable_type and target are provided so target should be ignored.
    rng = np.random.default_rng(_nprs)

    N = 9999
    dic = {
        "latitude": rng.random(N) * 9 + 42.0,
        "altitude": rng.random(N) * 1600.0,
        "azimuth": rng.random(N) * 360.0,
        "month": rng.integers(1, 13, N),
        "day": rng.integers(1, 31, N),
        "hour": rng.integers(0, 24, N),
        "ambient_temperature": rng.random(N) * 30.0,
        "wind_speed": rng.random(N) * 7.0,
        "wind_angle": rng.random(N) * 90.0,
        "core_diameter": rng.integers(2, size=N)
        * solver.default_values()["core_diameter"],
    }

    cable_type = rng.choice([CableType.BIMETALLIC, CableType.HOMOGENEOUS], size=N)

    target = np.array([TargetType.CORE] * N)

    for s in _solvers(dic):
        result_with_specified_cable_type = s.steady_intensity(
            max_conductor_temperature=100.0,
            cable_type=cable_type,
            target=target,
            return_err=True,
            return_power=True,
        )

        result_with_specified_target = s.steady_intensity(
            max_conductor_temperature=100.0,
            cable_type=cable_type,
            return_err=True,
            return_power=True,
        )

        assert np.allclose(
            result_with_specified_cable_type[VariableType.TRANSIT.value],
            result_with_specified_target[VariableType.TRANSIT.value],
        )
