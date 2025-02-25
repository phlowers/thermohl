import numpy as np
import pytest

from thermohl.power.cigre import SolarHeating


def test_solar_radiation_scalar():
    lat = 45.0
    azm = 180.0
    albedo = 0.2
    month = 6
    day = 21
    hour = 12.0
    expected = 1309.2

    result = SolarHeating._solar_radiation(lat, azm, albedo, month, day, hour)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_radiation_array():
    lat = np.array([45.0, 50.0, 55.0])
    azm = np.array([180.0, 180.0, 180.0])
    albedo = np.array([0.2, 0.2, 0.2])
    month = np.array([6, 6, 6])
    day = np.array([21, 21, 21])
    hour = np.array([12.0, 12.0, 12.0])
    expected = np.array([1309.2, 1267.965, 0.0])

    result = SolarHeating._solar_radiation(lat, azm, albedo, month, day, hour)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_radiation_default_albedo():
    lat = np.array([45.0, 50.0, 55.0])
    azm = np.array([180.0, 180.0, 180.0])
    albedo = 0.2
    month = np.array([6, 6, 6])
    day = np.array([21, 21, 21])
    hour = np.array([12.0, 12.0, 12.0])
    expected = np.array([1309.2, 1267.965, 0.0])

    result = SolarHeating._solar_radiation(lat, azm, albedo, month, day, hour)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_radiation_mismatched_array_sizes():
    lat = np.array([45.0, 50.0])
    azm = np.array([180.0, 180.0, 180.0])
    albedo = np.array([0.2, 0.2])
    month = np.array([6, 6, 6])
    day = np.array([21, 21])
    hour = np.array([12.0, 12.0])
    with pytest.raises(ValueError):
        SolarHeating._solar_radiation(lat, azm, albedo, month, day, hour)


solar_heating_instances = [
    (
        SolarHeating(
            lat=np.array([45.0, 50.0, 55.0]),
            azm=np.array([180.0, 180.0, 180.0]),
            al=np.array([0.2, 0.2, 0.2]),
            month=np.array([6, 6, 6]),
            day=np.array([21, 21, 21]),
            hour=np.array([12.0, 12.0, 12.0]),
            D=np.array([0.01, 0.01, 0.01]),
            alpha=np.array([0.9, 0.9, 0.9]),
        ),
        np.array([12.3978, 11.8759, 11.2545]),
    ),
    (
        SolarHeating(
            lat=45.0,
            azm=180.0,
            al=0.2,
            month=6,
            day=21,
            hour=12.0,
            D=0.01,
            alpha=0.9,
        ),
        12.3978,
    ),
]


@pytest.mark.parametrize(
    "solar_heating, expected",
    solar_heating_instances,
    ids=[
        "SolarHeating with arrays",
        "SolarHeating with scalars",
    ],
)
def test_solar_heating_value_scalar(solar_heating, expected):
    T = 25.0

    result = solar_heating.value(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_heating_value_array():
    lat = np.array([45.0, 50.0, 55.0])
    azm = np.array([180.0, 180.0, 180.0])
    albedo = np.array([0.2, 0.2, 0.2])
    month = np.array([6, 6, 6])
    day = np.array([21, 21, 21])
    hour = np.array([12.0, 12.0, 12.0])
    D = np.array([0.01, 0.01, 0.01])
    alpha = np.array([0.9, 0.9, 0.9])
    T = np.array([25.0, 30.0, 35.0])
    solar_heating = SolarHeating(lat, azm, albedo, month, day, hour, D, alpha)
    expected = np.array([12.3978, 11.8759, 11.2545])

    result = solar_heating.value(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_heating_value_mismatched_array_sizes_should_raise_error():
    lat = np.array([45.0, 50.0])
    azm = np.array([180.0, 180.0, 180.0])
    albedo = np.array([0.2, 0.2])
    month = np.array([6, 6, 6])
    day = np.array([21, 21])
    hour = np.array([12.0, 12.0])
    D = np.array([0.01, 0.01])
    alpha = np.array([0.9, 0.9])
    T = np.array([25.0, 30.0])
    with pytest.raises(ValueError):
        solar_heating = SolarHeating(lat, azm, albedo, month, day, hour, D, alpha)
        solar_heating.value(T)


@pytest.mark.parametrize(
    "solar_heating, expected",
    solar_heating_instances,
    ids=[
        "SolarHeating with arrays",
        "SolarHeating with scalars",
    ],
)
def test_solar_heating_derivative_temperature_scalar(solar_heating, expected):
    conductor_temperature = 25.0
    expected = np.zeros_like(conductor_temperature)

    result = solar_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "solar_heating, expected",
    solar_heating_instances,
    ids=[
        "SolarHeating with arrays",
        "SolarHeating with scalars",
    ],
)
def test_solar_heating_derivative_temperature_array(solar_heating, expected):
    conductor_temperature = np.array([25.0, 30.0, 35.0])
    expected = np.zeros_like(conductor_temperature)

    result = solar_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_solar_heating_derivative_mismatched_array_sizes_should_raise_error():
    lat = np.array([45.0, 50.0])
    azm = np.array([180.0, 180.0, 180.0])
    albedo = np.array([0.2, 0.2])
    month = np.array([6, 6, 6])
    day = np.array([21, 21])
    hour = np.array([12.0, 12.0])
    D = np.array([0.01, 0.01])
    alpha = np.array([0.9, 0.9])
    conductor_temperature = np.array([25.0, 30.0])
    with pytest.raises(ValueError):
        solar_heating = SolarHeating(lat, azm, albedo, month, day, hour, D, alpha)
        solar_heating.derivative(conductor_temperature)
