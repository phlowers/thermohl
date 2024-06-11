"""Models to compute equilibrium temperature or max intensity in a conductor."""

from thermohl.power import cigre as cigrep
from thermohl.power import cner as cnerp
from thermohl.power import ieee as ieeep
from thermohl.power import olla as ollap

from thermohl.solver.base import Args, Solver
from thermohl.solver.slv1d import Solver1D
from thermohl.solver.slv1t import Solver1T
from thermohl.solver.slv3t import Solver3T


def _factory(dic=None, heateq='1t', models='ieee'):
    if heateq == '1t':
        solver = Solver1T
    elif heateq == '3t':
        solver = Solver3T
    elif heateq == '1d':
        solver = Solver1D
    else:
        raise ValueError()

    if models == 'cigre':
        return solver(dic, cigrep.JouleHeating, cigrep.SolarHeating,
                      cigrep.ConvectiveCooling, cigrep.RadiativeCooling)
    elif models == 'ieee':
        return solver(dic, ieeep.JouleHeating, ieeep.SolarHeating,
                      ieeep.ConvectiveCooling, ieeep.RadiativeCooling)
    elif models == 'olla':
        return solver(dic, ollap.JouleHeatingMulti, ollap.SolarHeating,
                      ollap.ConvectiveCooling, ollap.RadiativeCooling)
    elif models == 'cner':
        return solver(dic, cnerp.JouleHeating, cnerp.SolarHeating,
                      cnerp.ConvectiveCooling, cnerp.RadiativeCooling)
    else:
        raise ValueError()


def cigre(dic=None, heateq='1t'):
    """Get a Solver using CIGRE models.

    Parameters
    ----------
    dic : dict, optional
        Input values. The default is None.
    heateq : str, optional
        Input heat equation.

    """
    return _factory(dic, heateq=heateq, models='cigre')


def ieee(dic=None, heateq='1t'):
    """Get a Solver using IEEE models.

    Parameters
    ----------
    dic : dict, optional
        Input values. The default is None.
    heateq : str, optional
        Input heat equation.

    """
    return _factory(dic, heateq=heateq, models='ieee')


def olla(dic=None, heateq='1t'):
    """Get a Solver using RTE-olla models.

    Parameters
    ----------
    dic : dict, optional
        Input values. The default is None.
    heateq : str, optional
        Input heat equation.

    """
    return _factory(dic, heateq=heateq, models='olla')


def cner(dic=None, heateq='1t'):
    """Get a Solver using RTE-cner models.

    Parameters
    ----------
    dic : dict, optional
        Input values. The default is None.
    heateq : str, optional
        Input heat equation.

    """
    return _factory(dic, heateq=heateq, models='cner')
