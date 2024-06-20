import numpy as np

import thermohl.power.ieee
from thermohl import solver


def test_power():
    """Compare computed values to hard-coded ones from ieee guide [find ref]."""
    dic = solver.default_values()

    dic['ws'] = 0.61
    dic['wa'] = 0.
    dic['epsilon'] = 0.8
    dic['alpha'] = 0.8
    dic['Ta'] = 40.
    # dic['Tmax'] =100.
    dic['Thigh'] = 75.
    dic['TLow'] = 25.
    dic['RDCHigh'] = 8.688E-05
    dic['RDCLow'] = 7.283E-05
    dic['azm'] = 90.
    dic['lat'] = 30.
    dic['tb'] = 1.
    dic['alt'] = 0.
    dic['D'] = 28.14 * 1.0E-03
    dic['d'] = 10.4 * 1.0E-03
    dic['month'] = 6
    dic['day'] = 10
    dic['hour'] = 11.
    # dic[''] =

    T = 100.

    assert np.isclose(thermohl.power.ieee.ConvectiveCooling(**dic).value(T), 81.93, rtol=0.002)
    assert np.isclose(thermohl.power.ieee.RadiativeCooling(**dic).value(T), 39.1, rtol=0.001)
    # assert np.isclose(thermohl.power.ieee.SolarHeating(**dic).value(T), 22.44, rtol=0.002)
