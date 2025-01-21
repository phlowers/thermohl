import numpy as np
import pytest
from thermohl.power.cner import JouleHeating, ConvectiveCooling


def test_rdc():
    I = np.array([10.0])
    D = np.array([0.01])
    d = np.array([0.005])
    A = np.array([0.0001])
    a = np.array([0.00005])
    km = np.array([1.0])
    ki = np.array([0.1])
    kl = np.array([0.004])
    kq = np.array([0.0001])
    RDC20 = np.array([0.02])
    T20 = 20.0
    f = 50.0
    T = np.array([30.0])

    joule_heating = JouleHeating(I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)
    result = joule_heating._rdc(T)

    expected_rdc = RDC20 * (1.0 + kl * (T - T20) + kq * (T - T20) ** 2)

    np.testing.assert_allclose(result, expected_rdc, rtol=1e-5)


def test_ks():
    I = np.array([10.0])
    D = np.array([0.01])
    d = np.array([0.005])
    A = np.array([0.0001])
    a = np.array([0.00005])
    km = np.array([1.0])
    ki = np.array([0.1])
    kl = np.array([0.004])
    kq = np.array([0.0001])
    RDC20 = np.array([0.02])
    T20 = 20.0
    f = 50.0
    T = np.array([30.0])

    joule_heating = JouleHeating(I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)
    rdc = joule_heating._rdc(T)
    result = joule_heating._ks(rdc)

    z = 8 * np.pi * f * (D - d) ** 2 / ((D**2 - d**2) * 1.0e07 * rdc)
    a = 7 * z**2 / (315 + 3 * z**2)
    b = 56 / (211 + z**2)
    beta = 1.0 - d / D
    expected_ks = 1.0 + a * (1.0 - 0.5 * beta - b * beta**2)

    np.testing.assert_allclose(result, expected_ks, rtol=1e-5)


def test_kem():
    I = np.array([10.0])
    D = np.array([0.01])
    d = np.array([0.005])
    A = np.array([0.0001])
    a = np.array([0.00005])
    km = np.array([1.0])
    ki = np.array([0.1])
    kl = np.array([0.004])
    kq = np.array([0.0001])
    RDC20 = np.array([0.02])
    T20 = 20.0
    f = 50.0

    joule_heating = JouleHeating(I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)
    result = joule_heating._kem(A, a, km, ki)

    s = (
        np.ones_like(I)
        * np.ones_like(A)
        * np.ones_like(a)
        * np.ones_like(km)
        * np.ones_like(ki)
    )
    z = s.shape == ()
    if z:
        s = np.array([1.0])
    I_ = I * s
    a_ = a * s
    A_ = A * s
    m = a_ > 0.0
    ki_ = ki * s
    kem = km * s
    kem[m] += ki_[m] * I_[m] / ((A_[m] - a_[m]) * 1.0e06)
    if z:
        kem = kem[0]

    np.testing.assert_allclose(result, kem, rtol=1e-5)


def test_joule_heating_value():
    I = np.array([10.0])
    D = np.array([0.01])
    d = np.array([0.005])
    A = np.array([0.0001])
    a = np.array([0.00005])
    km = np.array([1.0])
    ki = np.array([0.1])
    kl = np.array([0.004])
    kq = np.array([0.0001])
    RDC20 = np.array([0.02])
    T20 = 20.0
    f = 50.0
    T = np.array([30.0])

    joule_heating = JouleHeating(I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)
    result = joule_heating.value(T)

    expected_rdc = RDC20 * (1.0 + kl * (T - T20) + kq * (T - T20) ** 2)
    expected_ks = 1.0 + (
        7
        * (8 * np.pi * f * (D - d) ** 2 / ((D**2 - d**2) * 1.0e07 * expected_rdc)) ** 2
        / (
            315
            + 3
            * (8 * np.pi * f * (D - d) ** 2 / ((D**2 - d**2) * 1.0e07 * expected_rdc))
            ** 2
        )
    ) * (
        1.0
        - 0.5 * (1.0 - d / D)
        - 56
        / (
            211
            + (8 * np.pi * f * (D - d) ** 2 / ((D**2 - d**2) * 1.0e07 * expected_rdc))
            ** 2
        )
        * (1.0 - d / D) ** 2
    )
    expected_rac = joule_heating.kem * expected_ks * expected_rdc
    expected_value = expected_rac * I**2

    np.testing.assert_allclose(result, expected_value, rtol=1e-5)


def test_convective_cooling_value():
    temperature_ambiant = 25.0
    altitude = 100.0
    temperature = np.array([30.0])
    external_diameter = np.array([0.01])
    alpha = np.array([0.5])
    azimuth = np.array([2])
    wind_speed = np.array([10])
    wind_angle = np.array([11])

    convective_cooling = ConvectiveCooling(
        Ta=temperature_ambiant,
        alt=altitude,
        D=external_diameter,
        azm=azimuth,
        ws=wind_speed,
        wa=wind_angle,
        alpha=alpha,
    )
    result = convective_cooling.value(temperature)

    Tf = 0.5 * (temperature + temperature_ambiant)
    Td = temperature - temperature_ambiant
    vm = (1.293 - 1.525e-04 * altitude + 6.38e-09 * altitude**2) / (1 + 0.00367 * Tf)
    expected_value = np.maximum(
        convective_cooling._value_forced(Tf, Td, vm),
        convective_cooling._value_natural(Td, vm),
    )

    np.testing.assert_allclose(result, expected_value, rtol=1e-5)
