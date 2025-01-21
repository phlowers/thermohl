import pytest
import numpy as np
from isort.identify import imports

import thermohl.power.ieee
from thermohl import sun, solver
from thermohl.power.ieee import (
    Air,
    JouleHeating,
    _SRad,
    SolarHeating,
    ConvectiveCooling,
    ConvectiveCoolingBase,
)


# Tests Class Air
def test_volumic_mass_scalar():
    Tc = 25.0
    alt = 1000.0
    expected = (1.293 - 1.525e-04 * alt + 6.379e-09 * alt**2) / (1.0 + 0.00367 * Tc)

    result = Air.volumic_mass(Tc, alt)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_array():
    Tc = np.array([25.0, 30.0])
    alt = np.array([1000.0, 2000.0])
    expected = (1.293 - 1.525e-04 * alt + 6.379e-09 * alt**2) / (1.0 + 0.00367 * Tc)

    result = Air.volumic_mass(Tc, alt)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_default_altitude():
    Tc = 25.0
    expected = (1.293 - 1.525e-04 * 0.0 + 6.379e-09 * 0.0**2) / (1.0 + 0.00367 * Tc)

    result = Air.volumic_mass(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_array_default_altitude():
    Tc = np.array([25.0, 30.0])
    expected = (1.293 - 1.525e-04 * 0.0 + 6.379e-09 * 0.0**2) / (1.0 + 0.00367 * Tc)

    result = Air.volumic_mass(Tc)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_scalar():
    Tc = 25.0
    expected = (1.458e-06 * (Tc + 273.0) ** 1.5) / (Tc + 383.4)

    result = Air.dynamic_viscosity(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_array():
    Tc = np.array([25.0, 30.0])
    expected = (1.458e-06 * (Tc + 273.0) ** 1.5) / (Tc + 383.4)

    result = Air.dynamic_viscosity(Tc)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_thermal_conductivity_scalar():
    Tc = 25.0
    expected = 2.424e-02 + 7.477e-05 * Tc - 4.407e-09 * Tc**2

    result = Air.thermal_conductivity(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_thermal_conductivity_array():
    Tc = np.array([25.0, 30.0])
    expected = 2.424e-02 + 7.477e-05 * Tc - 4.407e-09 * Tc**2

    result = Air.thermal_conductivity(Tc)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


# Tests Class JouleHeating
def test_c_scalar():
    TLow = 20.0
    THigh = 80.0
    RDCLow = 0.1
    RDCHigh = 0.2
    expected = (RDCHigh - RDCLow) / (THigh - TLow)

    result = JouleHeating._c(TLow, THigh, RDCLow, RDCHigh)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_c_array():
    TLow = np.array([20.0, 30.0])
    THigh = np.array([80.0, 90.0])
    RDCLow = np.array([0.1, 0.15])
    RDCHigh = np.array([0.2, 0.25])
    expected = (RDCHigh - RDCLow) / (THigh - TLow)

    result = JouleHeating._c(TLow, THigh, RDCLow, RDCHigh)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_c_mixed():
    TLow = 20.0
    THigh = np.array([80.0, 90.0])
    RDCLow = 0.1
    RDCHigh = np.array([0.2, 0.25])
    expected = (RDCHigh - RDCLow) / (THigh - TLow)

    result = JouleHeating._c(TLow, THigh, RDCLow, RDCHigh)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_rdc_scalar():
    I = 100.0
    TLow = 20.0
    THigh = 80.0
    RDCLow = 0.1
    RDCHigh = 0.2
    joule_heating = JouleHeating(I, TLow, THigh, RDCLow, RDCHigh)
    T = 50.0
    expected = RDCLow + joule_heating.c * (T - TLow)

    result = joule_heating._rdc(T)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_rdc_array():
    I = 100.0
    TLow = 20.0
    THigh = 80.0
    RDCLow = 0.1
    RDCHigh = 0.2
    joule_heating = JouleHeating(I, TLow, THigh, RDCLow, RDCHigh)
    T = np.array([30.0, 40.0, 50.0])
    expected = RDCLow + joule_heating.c * (T - TLow)

    result = joule_heating._rdc(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_rdc_mixed():
    I = 100.0
    TLow = np.array([20.0, 25.0])
    THigh = np.array([80.0, 85.0])
    RDCLow = np.array([0.1, 0.15])
    RDCHigh = np.array([0.2, 0.25])
    joule_heating = JouleHeating(I, TLow, THigh, RDCLow, RDCHigh)
    T = np.array([30.0, 35.0])
    expected = RDCLow + joule_heating.c * (T - TLow)

    result = joule_heating._rdc(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_value_scalar():
    I = 100.0
    TLow = 20.0
    THigh = 80.0
    RDCLow = 0.1
    RDCHigh = 0.2
    joule_heating = JouleHeating(I, TLow, THigh, RDCLow, RDCHigh)
    T = 50.0
    expected = (RDCLow + joule_heating.c * (T - TLow)) * I**2

    result = joule_heating.value(T)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_value_array():
    I = 100.0
    TLow = 20.0
    THigh = 80.0
    RDCLow = 0.1
    RDCHigh = 0.2
    joule_heating = JouleHeating(I, TLow, THigh, RDCLow, RDCHigh)
    T = np.array([30.0, 40.0, 50.0])
    expected = (RDCLow + joule_heating.c * (T - TLow)) * I**2

    result = joule_heating.value(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_value_mixed():
    I = 100.0
    TLow = np.array([20.0, 25.0])
    THigh = np.array([80.0, 85.0])
    RDCLow = np.array([0.1, 0.15])
    RDCHigh = np.array([0.2, 0.25])
    joule_heating = JouleHeating(I, TLow, THigh, RDCLow, RDCHigh)
    T = np.array([30.0, 35.0])
    expected = (RDCLow + joule_heating.c * (T - TLow)) * I**2

    result = joule_heating.value(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_derivative_scalar():
    I = 100.0
    TLow = 20.0
    THigh = 80.0
    RDCLow = 0.1
    RDCHigh = 0.2
    joule_heating = JouleHeating(I, TLow, THigh, RDCLow, RDCHigh)
    conductor_temperature = 50.0
    expected = joule_heating.c * I**2

    result = joule_heating.derivative(conductor_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_derivative_array():
    I = 100.0
    TLow = 20.0
    THigh = 80.0
    RDCLow = 0.1
    RDCHigh = 0.2
    joule_heating = JouleHeating(I, TLow, THigh, RDCLow, RDCHigh)
    conductor_temperature = np.array([30.0, 40.0, 50.0])
    expected = joule_heating.c * I**2 * np.ones_like(conductor_temperature)

    result = joule_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_derivative_mixed():
    I = 100.0
    TLow = np.array([20.0, 25.0])
    THigh = np.array([80.0, 85.0])
    RDCLow = np.array([0.1, 0.15])
    RDCHigh = np.array([0.2, 0.25])
    joule_heating = JouleHeating(I, TLow, THigh, RDCLow, RDCHigh)
    conductor_temperature = np.array([30.0, 35.0])
    expected = joule_heating.c * I**2 * np.ones_like(conductor_temperature)

    result = joule_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


# Tests Class SRad
def test_catm_scalar():
    clean = [
        -4.22391e01,
        +6.38044e01,
        -1.9220e00,
        +3.46921e-02,
        -3.61118e-04,
        +1.94318e-06,
        -4.07608e-09,
    ]
    indus = [
        +5.31821e01,
        +1.4211e01,
        +6.6138e-01,
        -3.1658e-02,
        +5.4654e-04,
        -4.3446e-06,
        +1.3236e-08,
    ]
    srad = _SRad(clean, indus)
    x = 30.0
    trb = 0.5
    omt = 1.0 - trb
    A = omt * clean[6] + trb * indus[6]
    B = omt * clean[5] + trb * indus[5]
    C = omt * clean[4] + trb * indus[4]
    D = omt * clean[3] + trb * indus[3]
    E = omt * clean[2] + trb * indus[2]
    F = omt * clean[1] + trb * indus[1]
    G = omt * clean[0] + trb * indus[0]
    expected = A * x**6 + B * x**5 + C * x**4 + D * x**3 + E * x**2 + F * x**1 + G

    result = srad.catm(x, trb)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_catm_array():
    clean = [
        -4.22391e01,
        +6.38044e01,
        -1.9220e00,
        +3.46921e-02,
        -3.61118e-04,
        +1.94318e-06,
        -4.07608e-09,
    ]
    indus = [
        +5.31821e01,
        +1.4211e01,
        +6.6138e-01,
        -3.1658e-02,
        +5.4654e-04,
        -4.3446e-06,
        +1.3236e-08,
    ]
    srad = _SRad(clean, indus)
    x = np.array([30.0, 40.0])
    trb = np.array([0.5, 0.7])
    omt = 1.0 - trb
    A = omt * clean[6] + trb * indus[6]
    B = omt * clean[5] + trb * indus[5]
    C = omt * clean[4] + trb * indus[4]
    D = omt * clean[3] + trb * indus[3]
    E = omt * clean[2] + trb * indus[2]
    F = omt * clean[1] + trb * indus[1]
    G = omt * clean[0] + trb * indus[0]
    expected = A * x**6 + B * x**5 + C * x**4 + D * x**3 + E * x**2 + F * x**1 + G

    result = srad.catm(x, trb)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_radiation_scalar():
    clean = [
        -4.22391e01,
        +6.38044e01,
        -1.9220e00,
        +3.46921e-02,
        -3.61118e-04,
        +1.94318e-06,
        -4.07608e-09,
    ]
    indus = [
        +5.31821e01,
        +1.4211e01,
        +6.6138e-01,
        -3.1658e-02,
        +5.4654e-04,
        -4.3446e-06,
        +1.3236e-08,
    ]
    srad = _SRad(clean, indus)
    lat = 45.0
    alt = 1000.0
    azm = 180.0
    trb = 0.5
    month = 6
    day = 21
    hour = 12.0
    sa = sun.solar_altitude(lat, month, day, hour)
    sz = sun.solar_azimuth(lat, month, day, hour)
    th = np.arccos(np.cos(sa) * np.cos(sz - azm))
    K = 1.0 + 1.148e-04 * alt - 1.108e-08 * alt**2
    Q = srad.catm(np.rad2deg(sa), trb)
    expected = K * Q * np.sin(th)
    expected = np.where(expected > 0.0, expected, 0.0)

    result = srad(lat, alt, azm, trb, month, day, hour)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_radiation_array():
    clean = [
        -4.22391e01,
        +6.38044e01,
        -1.9220e00,
        +3.46921e-02,
        -3.61118e-04,
        +1.94318e-06,
        -4.07608e-09,
    ]
    indus = [
        +5.31821e01,
        +1.4211e01,
        +6.6138e-01,
        -3.1658e-02,
        +5.4654e-04,
        -4.3446e-06,
        +1.3236e-08,
    ]
    srad = _SRad(clean, indus)
    lat = np.array([45.0, 50.0])
    alt = np.array([1000.0, 2000.0])
    azm = np.array([180.0, 190.0])
    trb = np.array([0.5, 0.7])
    month = np.array([6, 7])
    day = np.array([21, 22])
    hour = np.array([12.0, 13.0])
    sa = sun.solar_altitude(lat, month, day, hour)
    sz = sun.solar_azimuth(lat, month, day, hour)
    th = np.arccos(np.cos(sa) * np.cos(sz - azm))
    K = 1.0 + 1.148e-04 * alt - 1.108e-08 * alt**2
    Q = srad.catm(np.rad2deg(sa), trb)
    expected = K * Q * np.sin(th)
    expected = np.where(expected > 0.0, expected, 0.0)

    result = srad(lat, alt, azm, trb, month, day, hour)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


# Tests SolarHeatingBase
def test_solar_heating_value_scalar():
    lat = 45.0
    alt = 1000.0
    azm = 180.0
    tb = 0.5
    month = 6
    day = 21
    hour = 12.0
    D = 0.01
    alpha = 0.9
    srad = 800.0
    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)
    T = 50.0
    expected = alpha * srad * D

    result = solar_heating.value(T)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_heating_value_array():
    lat = np.array([45.0, 50.0])
    alt = np.array([1000.0, 2000.0])
    azm = np.array([180.0, 190.0])
    tb = np.array([0.5, 0.7])
    month = np.array([6, 7])
    day = np.array([21, 22])
    hour = np.array([12.0, 13.0])
    D = np.array([0.01, 0.02])
    alpha = np.array([0.9, 0.8])
    srad = np.array([800.0, 900.0])
    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)
    T = np.array([50.0, 60.0])
    expected = alpha * srad * D

    result = solar_heating.value(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_heating_value_mixed():
    lat = 45.0
    alt = 1000.0
    azm = 180.0
    tb = 0.5
    month = 6
    day = 21
    hour = 12.0
    D = 0.01
    alpha = 0.9
    srad = 800.0
    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)
    T = np.array([50.0, 60.0])
    expected = alpha * srad * D * np.ones_like(T)

    result = solar_heating.value(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_heating_derivative_scalar():
    lat = 45.0
    alt = 1000.0
    azm = 180.0
    tb = 0.5
    month = 6
    day = 21
    hour = 12.0
    D = 0.01
    alpha = 0.9
    srad = 800.0
    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)
    conductor_temperature = 50.0
    expected = 0.0

    result = solar_heating.derivative(conductor_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_heating_derivative_array():
    lat = np.array([45.0, 50.0])
    alt = np.array([1000.0, 2000.0])
    azm = np.array([180.0, 190.0])
    tb = np.array([0.5, 0.7])
    month = np.array([6, 7])
    day = np.array([21, 22])
    hour = np.array([12.0, 13.0])
    D = np.array([0.01, 0.02])
    alpha = np.array([0.9, 0.8])
    srad = np.array([800.0, 900.0])
    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)
    conductor_temperature = np.array([50.0, 60.0])
    expected = np.array([0.0, 0.0])

    result = solar_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_heating_derivative_mixed():
    lat = 45.0
    alt = 1000.0
    azm = 180.0
    tb = 0.5
    month = 6
    day = 21
    hour = 12.0
    D = 0.01
    alpha = 0.9
    srad = 800.0
    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)
    conductor_temperature = np.array([50.0, 60.0])
    expected = np.array([0.0, 0.0])

    result = solar_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


# Tests SolarHeating
def test_solar_heating_init_scalar():
    lat = 45.0
    alt = 1000.0
    azm = 180.0
    tb = 0.5
    month = 6
    day = 21
    hour = 12.0
    D = 0.01
    alpha = 0.9
    srad = 800.0

    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)

    assert solar_heating.alpha == alpha
    assert np.isclose(solar_heating.srad, srad)
    assert np.isclose(solar_heating.D, D)


def test_solar_heating_init_array():
    lat = np.array([45.0, 50.0])
    alt = np.array([1000.0, 2000.0])
    azm = np.array([180.0, 190.0])
    tb = np.array([0.5, 0.7])
    month = np.array([6, 7])
    day = np.array([21, 22])
    hour = np.array([12.0, 13.0])
    D = np.array([0.01, 0.02])
    alpha = np.array([0.9, 0.8])
    srad = np.array([800.0, 900.0])

    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)

    assert np.allclose(solar_heating.alpha, alpha)
    assert np.allclose(solar_heating.srad, srad)
    assert np.allclose(solar_heating.D, D)


def test_solar_heating_init_mixed():
    lat = 45.0
    alt = 1000.0
    azm = 180.0
    tb = 0.5
    month = 6
    day = 21
    hour = 12.0
    D = 0.01
    alpha = 0.9
    srad = np.array([800.0, 900.0])

    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)

    assert solar_heating.alpha == alpha
    assert np.allclose(solar_heating.srad, srad)
    assert np.isclose(solar_heating.D, D)


def test_solar_heating_init_no_srad():
    lat = 45.0
    alt = 1000.0
    azm = 180.0
    tb = 0.5
    month = 6
    day = 21
    hour = 12.0
    D = 0.01
    alpha = 0.9

    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha)

    assert solar_heating.alpha == alpha
    assert solar_heating.srad is not None
    assert np.isclose(solar_heating.D, D)


# Tests Class ConvectiveCoolingBase


# Tests Class RadiativeCooling
def test_radiative_cooling_value():
    dic = solver.default_values()
    dic["Ta"] = 40.0
    dic["D"] = 28.14 * 1.0e-03
    dic["epsilon"] = 0.8
    temp = 100.0
    radiative_cooling = thermohl.power.ieee.RadiativeCooling(**dic)
    expected_result = (
        17.8
        * dic["epsilon"]
        * dic["D"]
        * (((temp + 273.0) / 100.0) ** 4 - ((dic["Ta"] + 273.0) / 100.0) ** 4)
    )

    result = radiative_cooling.value(temp)

    assert np.isclose(result, expected_result, rtol=0.001)


def test_radiative_cooling_derivative():
    dic = solver.default_values()

    dic["Ta"] = 40.0
    dic["D"] = 28.14 * 1.0e-03
    dic["epsilon"] = 0.8

    temp = 100.0

    radiative_cooling = thermohl.power.ieee.RadiativeCooling(**dic)
    result = radiative_cooling.derivative(temp)

    expected_result = 4.0 * 1.78e-07 * dic["epsilon"] * dic["D"] * temp**3

    assert np.isclose(result, expected_result, rtol=0.001)
