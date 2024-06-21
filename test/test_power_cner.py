import numpy as np
import pandas as pd

from thermohl.power import cner
from thermohl.solver import Args


class ExcelSheet:
    """Object to compare power terms from cner's excel sheet version 7."""

    def __init__(self, dic):
        self.args = Args(dic)
        self.args.nbc = dic['nbc']

    def joule_heating(self, Ta, I=None):
        if I is None:
            I = self.args['I']
        d = self.args['d']
        D = self.args['D']
        Rdc = self.args['RDC20'] * (1. + self.args['kl'] * (Ta - 20.) + self.args['kq'] * (Ta - 20.)**2)
        z = 8 * np.pi * 50. * (D - d)**2 / ((D**2 - d**2) * 1.0E+07 * Rdc)
        a = 7 * z**2 / (315 + 3 * z**2)
        b = 56 / (211 + z**2)
        beta = 1. - d / D
        kep = 1 + a * (1. - 0.5 * beta - b * beta**2)
        kem = np.where(
            (d > 0.) & (self.args['nbc'] == 3),
            self.args['km'] + self.args['ki'] * I / (self.args['A'] - self.args['a']) * 1.0E-06,
            1.
        )
        Rac = Rdc * kep * kem
        return Rac * self.args['I']**2

    def solar_heating(self):
        csm = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        O4 = csm[self.args['month'] - 1] + self.args['day']
        O5 = self.args['hour']
        O6 = np.deg2rad(self.args['lat'])
        Q4 = np.deg2rad(23.46 * np.sin(np.deg2rad((284 + O4) / 365 * 360)))
        Q5 = np.deg2rad((O5 - 12) * 15)
        Q6 = np.rad2deg(np.arcsin(np.cos(O6) * np.cos(Q4) * np.cos(Q5) + np.sin(O6) * np.sin(Q4)))
        q = np.maximum(
            -42 + 63.8 * Q6 - 1.922 * Q6**2 + 0.03469 * Q6**3 -
            0.000361 * Q6**4 + 0.000001943 * Q6**5 - 0.00000000408 * Q6**6,
            0.
        )
        Q7 = np.sin(Q5) / (np.sin(O6) * np.cos(Q5) - np.cos(O6) * np.tan(Q4))
        Q8 = np.deg2rad(180 + np.rad2deg(np.arctan(Q7)))
        O7 = np.pi / 2
        O2 = np.arccos(np.cos(np.deg2rad(Q6)) * np.cos(Q8 - O7))
        q *= np.sin(O2)
        return q * self.args['D'] * self.args['alpha']

    def convective_cooling(self, Ts):
        D = self.args['D']
        Tf = 0.5 * (Ts + self.args['Ta'])
        lm = 0.02424 + 0.00007477 * Tf - 0.000000004407 * Tf**2
        rho = (1.293 - 0.0001525 * self.args['alt'] + 0.00000000638 * self.args['alt']**2) / (1 + 0.00367 * Tf)
        mu = (0.000001458 * (Tf + 273)**1.5) / (Tf + 383.4)
        Re = self.args['ws'] * self.args['D'] * rho / mu
        F = np.maximum(1.01 + 1.35 * Re**0.52, 0.754 * Re**0.6)
        wa = np.deg2rad(self.args['wa'])
        K = 1.194 - np.cos(wa) + 0.194 * np.cos(2 * wa) + 0.368 * np.sin(2 * wa)
        PCn = 3.645 * rho**0.5 * D**0.75 * np.sign(Ts - self.args['Ta']) * np.abs(Ts - self.args['Ta'])**1.25
        PCf = F * lm * K * (Ts - self.args['Ta'])
        # print(f"re={Re}, kp={K}, lam={lm}")
        # print(f"pcn={PCn}, pcf={PCf}")
        return np.maximum(PCn, PCf)

    def radiative_cooling(self, Ts):
        D = self.args['D']
        return 17.8 * D * self.args['epsilon'] * (((273 + Ts) / 100)**4 - ((273 + self.args['Ta']) / 100)**4)


def excel_conductor_data():
    """Get conductor data from excel sheet (hard-coded)."""
    df = pd.DataFrame(
        dict(
            conductor=['ACSS1317', 'Aster228', 'Aster570', 'Crocus412', 'Pastel228', 'Petunia612'],
            D=[44., 19.6, 31.06, 26.4, 19.6, 32.1],
            d=[21.28, 0., 0., 12., 8.4, 13.25],
            A=[1317, 228, 570, 412, 228, 612],
            a=[0, 0, 0, 0, 0, 0],
            B=[1049, 228, 570, 323, 185, 508],
            RDC20=[0.0272, 0.146, 0.0583, 0.089, 0.18, 0.0657],
            kl=[0.004, 0.0036, 0.0036, 0.004, 0.0036, 0.0036],
            km=[1.006, 1., 1., 1., 1., 1.006],
            ki=[0.016, 0., 0., 0., 0., 0.016],
            kq=[8.E-07, 8.E-07, 8.E-07, 8.E-07, 8.E-07, 8.E-07],
            nbc=[3, 0, 0, 2, 2, 3],
        )
    )

    df['a'] = df['A'] - df['B']
    df.drop(columns=['B'], inplace=True)
    df['D'] *= 1.0E-03
    df['d'] *= 1.0E-03
    df['A'] *= 1.0E-06
    df['a'] *= 1.0E-06
    df['RDC20'] *= 1.0E-03

    return df


def scenarios():
    """Get list of hard-coded scenarios to test."""
    dic = dict(
        conductor=["Aster228", "Pastel228", "Petunia612", "Petunia612", "ACSS1317", "ACSS1317", "Aster228", "Aster228",
                   "Aster228", "Aster228"],
        Ta=[20., 20., 20., 20., 20., 20., 20., 20., 20., 20.],
        ws=[3., 3., 3., 0., 0., 3., 0.6, 0.6, 0.6, 0.6],
        wa=[90., 90., 90., 45., 45., 90., 90., 90., 90., 90.],
        lat=[46., 46., 46., 46., 46., 46., 46., 46., 46., 46.],
        alt=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        azm=90.,
        I=[1000., 1000., 1800., 1100., 3000., 4000., 700., 700., 700., 700.],
        alpha=0.9,
        epsilon=0.8,
        tb=0.,
        month=[3, 3, 3, 3, 3, 3, 3, 6, 6, 6],
        day=[7, 7, 7, 7, 7, 7, 7, 21, 21, 21],
        hour=[0., 0., 0., 0., 0., 0., 12., 12., 19., 12.],
    )
    df = pd.DataFrame(dic)
    dg = excel_conductor_data()
    df = pd.merge(df, dg, on='conductor', how='left').drop(columns='conductor')
    return df


def test_compare_power():
    """Compare computed values to hard-coded ones from ieee guide [find ref]."""

    T = np.linspace(-50, +250, 999)

    ds = scenarios()
    n = len(ds)
    ds = pd.concat(len(T) * (ds,)).reset_index(drop=True)
    T = np.concatenate([n * (t,) for t in T])

    from thermohl.utils import df2dct
    d1 = df2dct(ds)
    ds['wa'] = np.rad2deg(np.arcsin(np.sin(np.deg2rad(np.abs(ds['azm'] - ds['wa']) % 180.))))
    d2 = df2dct(ds)
    del (ds, n)

    pj = cner.JouleHeating(**d1)
    ps = cner.SolarHeating(**d1)
    pc = cner.ConvectiveCooling(**d1)
    pr = cner.RadiativeCooling(**d1)
    ex = ExcelSheet(d2)

    assert np.all(np.isclose(ex.joule_heating(T), pj.value(T)))
    assert np.all(np.isclose(ex.solar_heating(), ps.value(0.)))
    assert np.all(np.isclose(ex.convective_cooling(T), pc.value(T)))
    assert np.all(np.isclose(ex.radiative_cooling(T), pr.value(T)))
