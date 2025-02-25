import numpy as np

from thermohl.power.cner import Air


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
