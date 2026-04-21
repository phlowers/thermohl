# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl import solver
from thermohl.solver import HeatEquationType, ModelType
from thermohl.solver.entities import (
    TargetType,
    TemperatureType,
    VariableType,
)


def _solvers():
    li = []
    for heat_equation in [
        HeatEquationType.ONE_TEMPERATURE,
        HeatEquationType.THREE_TEMPERATURES,
    ]:
        for m in [
            ModelType.RTE,
            ModelType.CIGRE,
            ModelType.IEEE,
            ModelType.OLLA,
        ]:
            li.append(solver._factory(dic=None, heat_equation=heat_equation, model=m))
    return li


def _ampargs(s: solver.Solver, t: dict[str, np.array]):
    if isinstance(s, solver.Solver1T):
        a = {"max_conductor_temperature": t[VariableType.TEMPERATURE.value]}
    elif isinstance(s, solver.Solver3T):
        a = {
            "max_conductor_temperature": t[TemperatureType.SURFACE.value],
            "target": TargetType.SURFACE,
        }
    else:
        raise NotImplementedError
    return a


def _traargs(s: solver.Solver, ds: dict[str, np.array], t):
    if isinstance(s, solver.Solver1T):
        a = {"offset": t, "T0": ds[VariableType.TEMPERATURE.value]}
    elif isinstance(s, solver.Solver3T):
        a = {
            "offset": t,
            "surface_temperature_0": ds[TemperatureType.SURFACE.value],
            "core_temperature_0": ds[TemperatureType.CORE.value],
        }
    else:
        raise NotImplementedError
    return a


def test_power_default():
    """Check that PowerTerm.value(x) returns correct shape depending on init dict and temperature input."""
    for s in _solvers():
        for p in [
            s.joule_heating,
            s.solar_heating,
            s.convective_cooling,
            s.radiative_cooling,
            s.precipitation_cooling,
        ]:
            p.__init__(**s.args.__dict__)
            assert np.isscalar(p.value(0.0))
            assert p.value(np.array([0.0])).shape == (1,)
            assert p.value(np.array([0.0, 10.0])).shape == (2,)


def test_power_1d():
    """Check that PowerTerm.value(x) returns correct shape depending on init dict and temperature input."""
    n = 61
    for s in _solvers():
        d = s.args.__dict__.copy()
        d[VariableType.TRANSIT.value] = np.linspace(0.0, +999.0, n)
        d["solar_absorptivity"] = np.linspace(0.5, 0.9, n)
        d["ambient_temperature"] = np.linspace(-10.0, +50.0, n)
        for p in [
            s.joule_heating,
            s.solar_heating,
            s.convective_cooling,
            s.radiative_cooling,
            s.precipitation_cooling,
        ]:
            p.__init__(**d)
            v = p.value(0.0)
            assert np.isscalar(v) or v.shape == (n,)
            v = p.value(np.array([0.0]))
            assert v.shape == (1,) or v.shape == (n,)
            assert p.value(np.linspace(-10, +50, n)).shape == (n,)


def test_steady_default():
    for _solver in _solvers():
        temperature = _solver.steady_temperature()
        a = _ampargs(_solver, temperature)
        intensity = _solver.steady_intensity(**a)
        assert len(list(temperature.values())[0]) == 1
        assert len(list(intensity.values())[0]) == 1


def test_steady_1d():
    n = 61
    for _solver in _solvers():
        _solver.args.ambient_temperature = np.linspace(-10, +50, n)
        _solver.update()
        temperature = _solver.steady_temperature()
        a = _ampargs(_solver, temperature)
        intensity = _solver.steady_intensity(**a)
        assert len(list(temperature.values())[0]) == n
        assert len(list(intensity.values())[0]) == n


def test_steady_1d_mix():
    n = 61
    for _solver in _solvers():
        _solver.args.ambient_temperature = np.linspace(-10, +50, n)
        _solver.args.transit = np.array([199.0])
        _solver.update()
        temperature = _solver.steady_temperature()
        a = _ampargs(_solver, temperature)
        intensity = _solver.steady_intensity(**a)
        assert len(list(temperature.values())[0]) == n
        assert len(list(intensity.values())[0]) == n


def test_transient_0():
    for _solver in _solvers():
        time = np.linspace(0, 3600, 361)

        steady_temperature = _solver.steady_temperature()
        transient_temperature_args = _traargs(_solver, steady_temperature, time)

        result_without_power = _solver.transient_temperature(
            **transient_temperature_args
        )
        assert len(result_without_power.pop(VariableType.TIME.value)) == len(time)
        for key, value in result_without_power.items():
            if key.startswith("input_"):
                continue
            assert value.shape == (len(time),)

        result_with_power = _solver.transient_temperature(
            **{**transient_temperature_args, "return_power": True}
        )

        assert len(result_with_power.pop(VariableType.TIME.value)) == len(time)
        for key, value in result_with_power.items():
            if key.startswith("input_"):
                continue
            assert value.shape == (len(time),), value


def test_transient_1():
    n = 7
    for _solver in _solvers():
        _solver.args.ambient_temperature = np.linspace(-10, +50, n)
        _solver.update()

        time = np.linspace(0, 3600, 361)

        steady_temperature = _solver.steady_temperature()
        transient_temperature_args = _traargs(_solver, steady_temperature, time)

        result_without_power = _solver.transient_temperature(
            **transient_temperature_args
        )
        assert len(result_without_power.pop(VariableType.TIME.value)) == len(time)
        for key, value in result_without_power.items():
            if key.startswith("input_"):
                continue
            assert value.shape == (len(time), n)

        result_with_power = _solver.transient_temperature(
            **{**transient_temperature_args, "return_power": True}
        )
        assert len(result_with_power.pop(VariableType.TIME.value)) == len(time)
        for key, value in result_with_power.items():
            if key.startswith("input_"):
                continue
            assert value.shape == (len(time), n)
