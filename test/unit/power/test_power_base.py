import pytest
import numpy as np

from thermohl.power.base import PowerTerm, RadiativeCooling


# Tests for PowerTerm
def test_power_term_value_with_scalar():
    power_term = PowerTerm()
    T = 25.0

    result = power_term.value(T)

    assert result == 0.0


def test_power_term_value_with_array():
    power_term = PowerTerm()
    T = np.array([25.0, 30.0, 35.0])
    expected = np.array([0.0, 0.0, 0.0])

    result = power_term.value(T)

    np.testing.assert_array_equal(result, expected)


def test_power_term_derivative_with_scalar():
    power_term = PowerTerm()
    temperature = 25.0

    result = power_term.derivative(temperature)

    assert result == 0.0


def test_power_term_derivative_with_array():
    power_term = PowerTerm()
    temperatures = np.array([25.0, 30.0, 35.0])
    expected = np.array([0.0, 0.0, 0.0])

    result = power_term.derivative(temperatures)

    np.testing.assert_array_equal(result, expected)


# Tests for RadiativeCooling.value
def test_radiative_cooling_value_with_scalar():
    Ta = 20.0
    D = 0.1
    epsilon = 0.9
    T = 25.0
    radiative_cooling = RadiativeCooling(Ta, D, epsilon)

    result = radiative_cooling.value(T)

    expected = (
        np.pi
        * radiative_cooling.sigma
        * epsilon
        * D
        * ((T + radiative_cooling.zerok) ** 4 - (Ta + radiative_cooling.zerok) ** 4)
    )
    assert result == pytest.approx(expected)


def test_radiative_cooling_value_with_array():
    Ta = 20.0
    D = 0.1
    epsilon = 0.9
    T = np.array([25.0, 30.0, 35.0])
    radiative_cooling = RadiativeCooling(Ta, D, epsilon)

    result = radiative_cooling.value(T)

    expected = (
        np.pi
        * radiative_cooling.sigma
        * epsilon
        * D
        * ((T + radiative_cooling.zerok) ** 4 - (Ta + radiative_cooling.zerok) ** 4)
    )
    np.testing.assert_array_almost_equal(result, expected)


def test_radiative_cooling_derivative_with_scalar():
    Ta = 20.0
    D = 0.1
    epsilon = 0.9
    T = 25.0
    radiative_cooling = RadiativeCooling(Ta, D, epsilon)

    result = radiative_cooling.derivative(T)

    expected = 4.0 * np.pi * radiative_cooling.sigma * epsilon * D * T**3
    assert result == pytest.approx(expected)


def test_radiative_cooling_derivative_with_array():
    Ta = 20.0
    diameter = 0.1
    epsilon = 0.9
    T = np.array([25.0, 30.0, 35.0])
    radiative_cooling = RadiativeCooling(Ta, diameter, epsilon)

    result = radiative_cooling.derivative(T)

    expected = 4.0 * np.pi * radiative_cooling.sigma * epsilon * diameter * T**3
    np.testing.assert_array_almost_equal(result, expected)
