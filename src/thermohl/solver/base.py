"""Base class to build a solver for heat equation."""


class Solver:
    """Object to solve a temperature problem.

    The temperature of a conductor is driven by four power terms, two heating
    terms (joule and solar heating) and two cooling terms (convective and
    radiative cooling). This class is used to solve a temperature problem with
    the heating and cooling terms passed to its __init__ function.
    """

    def __init__(self):
        return

    def _rhs_value(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (self.jh.value(T, **self.dc) + self.sh.value(T, **self.dc) -
                self.cc.value(T, **self.dc) - self.rc.value(T, **self.dc) -
                self.pc.value(T, **self.dc))

    def steady_temperature(self):
        return

    def transient_temperature(self):
        return

    def steady_intensity(self):
        return
