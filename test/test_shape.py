import numpy as np

from thermohl import solver


def test_0():
    # todo: loop on heateq and models ..

    s = solver._factory(dic=None, heateq='1t', models='ieee')
    t = s.steady_temperature()
    i = s.steady_intensity(T=t['T'].values)

    assert len(t) == 1
    assert len(i) == 1


def test_1():
    # todo: loop on heateq and models ..

    n = 61
    s = solver._factory(dic=None, heateq='1t', models='ieee')
    s.args.Ta = np.linspace(-10, +50, n)
    s.update()
    t = s.steady_temperature()
    i = s.steady_intensity(T=t['T'].values)

    assert len(t) == n
    assert len(i) == n


def test_2():
    # todo: loop on heateq and models ..

    n = 61
    s = solver._factory(dic=None, heateq='1t', models='ieee')
    s.args.Ta = np.linspace(-10, +50, n)
    s.args.I = np.array([199.])
    s.update()
    t = s.steady_temperature()
    i = s.steady_intensity(T=t['T'].values)

    assert len(t) == n
    assert len(i) == n


def test_3():
    # todo: loop on heateq and models ..

    s = solver._factory(dic=None, heateq='1t', models='ieee')
    t = np.linspace(0, 3600, 361)
    I = 199 * np.ones_like(t)

    ds = s.steady_temperature()
    r = s.transient_temperature(t, T0=ds['T'].values[0], transit=I)
    r = s.transient_temperature(t, T0=ds['T'].values, transit=I, return_power=True)

    assert len(r['time']) == len(t)
    assert r['T'].shape == (len(t),)


def test_4():
    # todo: loop on heateq and models ..

    n = 7
    s = solver._factory(dic=None, heateq='1t', models='ieee')
    s.args.Ta = np.linspace(-10, +50, n)
    s.update()

    t = np.linspace(0, 3600, 361)
    I = 199 * np.ones_like(t)

    ds = s.steady_temperature()
    r = s.transient_temperature(t, T0=ds['T'].values, transit=I)
    r = s.transient_temperature(t, T0=ds['T'].values, transit=I, return_power=True)

    assert len(r['time']) == len(t)
    assert r['T'].shape == (len(t), s.args.max_len())
