from typing import Union

import numpy as np

from thermohl.power.base import PowerTerm


class JouleHeating(PowerTerm):

    def skin_effect_coefficient(self):
        pass

    def electromagnetic_effect_coefficient(self):
        pass

    def rac(self):
        pass

    def rdc(self):
        pass

    def value(self, T: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        pass
