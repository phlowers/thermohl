import numpy as np

from thermohl import solver

_nprs = 123456


def _solvers(dic=None):
    return [solver._factory(dic=dic, heateq='3t', model=m) for m in ['cner', 'cigre', 'ieee', 'olla']]


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
        # compute guess with 1t solver
        s1 = solver._factory(dic=dic, heateq='1t', model='ieee')
        t1 = s1.steady_temperature(tol=2., maxiter=16, return_err=False, return_power=False)
        t1 = t1['t'].values
        # 3t solve
        df = s.steady_temperature(Tsg=t1, Tcg=t1, return_err=True, return_power=True, tol=tol, maxiter=64)
        # checks
        assert np.all(df['err'] < tol)
        assert np.all(np.isclose(s.balance(ts=df['t_surf'], tc=df['t_core']).values, 0., atol=tol))
        assert np.all(np.isclose(s.morgan(ts=df['t_surf'], tc=df['t_core']).values, 0., atol=tol))


def test_consistency():
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
        d=np.random.randint(2, size=N) * solver.default_values()['d'],
    )

    for s in _solvers(dic):
        for t in ['surf', 'avg', 'core']:
            # solve intensity with different targets
            df = s.steady_intensity(Tmax=100., target=t, return_err=True, return_power=True, tol=1.0E-09, maxiter=64)
            assert np.all(df['err'] < tol)
            assert np.all(np.isclose(s.balance(ts=df['t_surf'], tc=df['t_core']).values, 0., atol=tol))
            assert np.all(np.isclose(s.morgan(ts=df['t_surf'], tc=df['t_core']).values, 0., atol=tol))
            # set args intensity to newly founds ampacities
            s.args['I'] = df['I'].values
            s.update()
            # compute guess with 1t solver
            s1 = solver._factory(dic=dic, heateq='1t', model='ieee')
            t1 = s1.steady_temperature(tol=5., maxiter=16, return_err=False, return_power=False)
            t1 = t1['t'].values
            # 3t solve
            dg = s.steady_temperature(Tsg=t1, Tcg=t1, return_err=True, return_power=True, tol=1.0E-09, maxiter=64)
            # check consistency
            assert np.all(np.isclose(dg['t_' + t].values, 100.))
