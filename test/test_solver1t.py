import numpy as np

from thermohl import solver

_nprs = 123456


def _solvers(dic=None):
    return [solver._factory(dic=dic, heateq='1t', model=m) for m in ['cner', 'cigre', 'ieee', 'olla']]


def test_balance():
    tol = 1.0E-09
    np.random.seed(_nprs)
    N = 9999
    dic = dict(
        lat=np.random.uniform(42., 51., N),
        alt=np.random.uniform(0., 1600., N),
        azm=np.random.uniform(0., 360., N),
        month=np.random.randint(1, 13, N),
        day=np.random.randint(1, 31, N),
        hour=np.random.randint(0, 24, N),
        Ta=np.random.uniform(0., 30., N),
        ws=np.random.uniform(0., 7., N),
        wa=np.random.uniform(0., 90., N),
        I=np.random.uniform(40., 4000., N),
        d=np.random.randint(2, size=N) * solver.default_values()['d'],
    )

    for s in _solvers(dic):
        df = s.steady_temperature(return_err=True, return_power=True, tol=tol, maxiter=64)
        assert np.all(df['err'] < tol)
        bl = np.abs(df['P_joule'] + df['P_solar'] - df['P_convection'] - df['P_radiation'] - df['P_precipitation'])
        atol = np.maximum(np.abs(s.balance(df['t'] + 0.5 * df['err'])),
                          np.abs(s.balance(df['t'] - 0.5 * df['err'])))
        assert np.all(np.isclose(bl, 0., atol=atol))


def test_consistency():
    np.random.seed(_nprs)
    N = 9999
    dic = dict(
        lat=np.random.uniform(42., 51., N),
        alt=np.random.uniform(0., 1600., N),
        azm=np.random.uniform(0., 360., N),
        month=np.random.randint(1, 13, N),
        day=np.random.randint(1, 31, N),
        hour=np.random.randint(0, 24, N),
        Ta=np.random.uniform(0., 30., N),
        ws=np.random.uniform(0., 7., N),
        wa=np.random.uniform(0., 90., N),
        d=np.random.randint(2, size=N) * solver.default_values()['d'],
    )

    for s in _solvers(dic):
        df = s.steady_intensity(T=100., return_err=True, return_power=True, tol=1.0E-09, maxiter=64)
        bl = df['P_joule'] + df['P_solar'] - df['P_convection'] - df['P_radiation'] - df['P_precipitation']
        assert np.all(np.isclose(bl, 0., atol=1.0E-06))
        s.args['I'] = df['I'].values
        s.update()
        dg = s.steady_temperature(return_err=True, return_power=True, tol=1.0E-09, maxiter=64)
        assert np.all(np.isclose(dg['t'].values, 100.))
