"""Power terms implementation using CIGRE recommendations.

See Thermal behaviour of overhead conductors, study committee 22, working
group 12, 2002.
"""

from .air import Air
from .solar_heating import SolarHeating
from .convective_cooling import ConvectiveCooling
from .joule_heating import JouleHeating
from .radiative_cooling import RadiativeCooling
