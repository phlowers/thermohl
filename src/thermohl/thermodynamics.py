"""Thermodynamics quantities. Values from Wikipedia unless specified."""
import numpy as np

# Standard temperature and pressure from EPA and NIST; _T0 in K, _p0 in Pa
_T0, _p0 = 293.15, 1.01325E+05

# Boltzman constant, Avogadro number and Gas constant (all in SI)
_kb = 1.380649E-23
_Na = 6.02214076E+23
_R = _kb * _Na


class Air:

    @staticmethod
    def heat_capacity(T=_T0):
        """In J.kg**-1.K**-1"""
        return np.interp(T, np.array([240., 600.]), np.array([1.006, 1.051]))


class Water:

    @staticmethod
    def boiling_point(p=_p0):
        """Using Clausius–Clapeyron equation; in K."""
        # convert H from J.kg**-1 to J.mol**-1 using molar mass
        H = Water.heat_of_vaporization() * 0.018015
        Tb = 1. / (1 / 373.15 - _R * np.log(p / _p0) / H)
        return Tb

    @staticmethod
    def heat_capacity(T=_T0):
        """From NIST webbook; in J.kg**-1.K**-1.
        See https://webbook.nist.gov/cgi/cbook.cgi?Name=Water&Units=SI.
        """
        A = -203.6060
        B = +1523.290
        C = -3196.413
        D = +2474.455
        E = 3.855326
        t = T / 1000.
        return A + B * t + C * t**2 + D * t**3 + E / t**2

    @staticmethod
    def heat_of_vaporization():
        # """At T=373.15 K and normal pressure; in J.mol**-1."""
        # return 4.0660E+04
        """At T=373.15 K and normal pressure; in J.kg**-1."""
        return 2.257E+05

    @staticmethod
    def vapor_pressure(T=_T0):
        """Using Buck equation; in Pa."""
        Tc = T - 273.15
        return 611.21 * np.exp((18.678 - Tc / 234.5) * (Tc / (257.14 + Tc)))

    @staticmethod
    def volumic_mass(T=_T0):
        """In kg.m**-3."""
        xp = np.array(
            [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.,
             23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 45., 50., 55.,
             60., 65., 70., 75., 80., 85., 90., 95., 100.]
        )
        fp = np.array(
            [999.840, 999.899, 999.940, 999.964, 999.972, 999.964, 999.940, 999.902, 999.848, 999.781, 999.700,
             999.6, 999.5, 999.38, 999.24, 999.1, 998.94, 998.77, 998.59, 998.4, 998.2, 997.99, 997.77, 997.54, 997.29,
             997.04, 996.78, 996.51, 996.23, 995.94, 995.64, 995.34, 995.02, 994.7, 994.37, 994.03, 993.68, 993.32,
             992.96, 992.59, 992.21, 990.21, 988.03, 985.69, 983.19, 980.55, 977.76, 974.84, 971.79, 968.61, 965.3,
             961.88, 958.35]
        )
        return np.interp(T, xp, fp)


class Ice:

    @staticmethod
    def heat_capacity():
        """In J.kg**-1.K**-1"""
        return 2.093E+03

    @staticmethod
    def heat_of_fusion():
        """At T=273.15 K and normal pressure; in J.kg**-1."""
        return 3.3355E+05
