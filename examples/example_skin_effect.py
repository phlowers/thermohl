#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from thermohl import rte
from thermohl import solver


def test_skin_effect():
    x = np.linspace(1., 50., 50)
    D = 1.0
    R = 60 / x**2 / 1609.34  # Rdc from x and mile to meter conversion
    ra = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.76,
                   0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92])
    dl = ra * D
    cl = cm.Spectral(np.linspace(0, 1, len(dl)))

    jh = rte.JouleHeating()

    plt.figure()
    for i, d in enumerate(dl):
        plt.plot(x, jh._ks(R, D, d, 50.), '-', c=cl[i], label='with $r_{in}/r_{out}$=%.2f' % (d,))
    plt.grid(True)
    plt.legend()
    plt.ylim([1.00, 1.15])
    plt.title('__ks__() [from RTE (Bessel approx)]')

    plt.show()

    return


if __name__ == '__main__':
    """Test skin effect coefficient."""

    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    test_skin_effect()
