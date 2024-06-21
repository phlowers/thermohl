import numpy as np

from thermohl import solver


def _solvers():
    return [solver._factory(dic=None, heateq='1t', model=m) for m in ['cner', 'cigre', 'ieee', 'olla']]


def test_power_default():
    """Check that PowerTerm.value(x) returns correct shape depending on init dict and temperature input."""
    for s in _solvers():
        for p in [s.jh, s.sh, s.cc, s.rc, s.pc]:
            p.__init__(**s.args.__dict__)
            assert np.isscalar(p.value(0.))
            assert p.value(np.array([0.])).shape == (1,)
            assert p.value(np.array([0., 10.])).shape == (2,)


def test_power_1d():
    """Check that PowerTerm.value(x) returns correct shape depending on init dict and temperature input."""
    n = 61
    for s in _solvers():
        d = s.args.__dict__.copy()
        d['I'] = np.linspace(0., +999., n)
        d['alpha'] = np.linspace(0.5, 0.9, n)
        d['Ta'] = np.linspace(-10., +50., n)
        for p in [s.jh, s.sh, s.cc, s.rc, s.pc]:
            p.__init__(**d)
            v = p.value(0.)
            assert np.isscalar(p.value(0.)) or p.value(0.).shape == (n,)
            v = p.value(np.array([0.]))
            assert v.shape == (1,) or v.shape == (n,)
            assert p.value(np.linspace(-10, +50, n)).shape == (n,)


def test_steady_default():
    for s in _solvers():
        t = s.steady_temperature()
        i = s.steady_intensity(T=t['t'].values)

        assert len(t) == 1
        assert len(i) == 1


def test_steady_1d():
    n = 61
    for s in _solvers():
        s.args.Ta = np.linspace(-10, +50, n)
        s.update()
        t = s.steady_temperature()
        i = s.steady_intensity(T=t['t'].values)

        assert len(t) == n
        assert len(i) == n


def test_steady_1d_mix():
    n = 61
    for s in _solvers():
        s.args.Ta = np.linspace(-10, +50, n)
        s.args.I = np.array([199.])
        s.update()
        t = s.steady_temperature()
        i = s.steady_intensity(T=t['t'].values)

        assert len(t) == n
        assert len(i) == n


def test_transient_0():
    for s in _solvers():
        t = np.linspace(0, 3600, 361)
        I = 199 * np.ones_like(t)

        ds = s.steady_temperature()
        r = s.transient_temperature(t, T0=ds['t'].values[0], transit=I)
        r = s.transient_temperature(t, T0=ds['t'].values, transit=I, return_power=True)

        assert len(r['time']) == len(t)
        assert r['T'].shape == (len(t),)


def test_transient_1():
    n = 7
    for s in _solvers():
        s.args.Ta = np.linspace(-10, +50, n)
        s.update()

        t = np.linspace(0, 3600, 361)
        I = 199 * np.ones_like(t)

        ds = s.steady_temperature()
        r = s.transient_temperature(t, T0=ds['t'].values, transit=I)
        r = s.transient_temperature(t, T0=ds['t'].values, transit=I, return_power=True)

        assert len(r['time']) == len(t)
        assert r['T'].shape == (len(t), s.args.max_len())
