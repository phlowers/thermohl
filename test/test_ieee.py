import numpy as np

import thermohl.power.ieee
from thermohl import solver


def test_compare_powers():
    """Compare computed values to hard-coded ones from ieee std 38-2012."""
    dic = solver.default_values()

    dic['ws'] = 0.61
    dic['wa'] = 0.
    dic['epsilon'] = 0.8
    dic['alpha'] = 0.8
    dic['Ta'] = 40.
    dic['THigh'] = 75.
    dic['TLow'] = 25.
    dic['RDCHigh'] = 8.688E-05
    dic['RDCLow'] = 7.283E-05
    dic['azm'] = 90.
    dic['lat'] = 30.
    dic['tb'] = 0.
    dic['alt'] = 0.
    dic['D'] = 28.14 * 1.0E-03
    dic['d'] = 10.4 * 1.0E-03
    dic['month'] = 6
    dic['day'] = 10
    dic['hour'] = 11.

    T = 100.

    assert np.isclose(thermohl.power.ieee.ConvectiveCooling(**dic).value(T), 81.93, rtol=0.002)
    assert np.isclose(thermohl.power.ieee.RadiativeCooling(**dic).value(T), 39.1, rtol=0.001)
    assert np.isclose(thermohl.power.ieee.SolarHeating(**dic).value(T), 22.44, rtol=0.001)
    jh = thermohl.power.ieee.JouleHeating(**dic)
    assert np.isclose(jh._rdc(T), 9.390E-05, rtol=1.0E-09)

    # additional debug
    thermohl.power.ieee.SolarHeating(**dic).value(T)

    from thermohl import sun

    sd = sun.solar_declination(dic['month'], dic['day'])
    assert np.isclose(np.rad2deg(sd), 23.0, rtol=0.001)

    ha = sun.hour_angle(dic['hour'], minute=0., second=0.)
    assert np.isclose(np.rad2deg(ha), -15.)

    sa = sun.solar_altitude(np.deg2rad(dic['lat']), dic['month'], dic['day'], dic['hour'])
    assert np.isclose(np.rad2deg(sa), 74.8, rtol=0.002)

    sz = sun.solar_azimuth(np.deg2rad(dic['lat']), dic['month'], dic['day'], dic['hour'])
    np.isclose(np.rad2deg(sz), 114., rtol=0.001)

    th = np.arccos(np.cos(sa) * np.cos(sz - dic['azm']))
    np.isclose(np.rad2deg(th), 76.1, rtol=0.02)
