import calendar
import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermohl import solver
from thermohl.power import cigre
from thermohl.power import cner
from thermohl.power import ieee
from thermohl.power import olla


def plot_joule_heating(dic):
    mdl = [
        dict(label='cigre', model=cigre.JouleHeating(**dic)),
        dict(label='ieee', model=ieee.JouleHeating(**dic)),
        dict(label='olla', model=olla.JouleHeating(**dic)),
        dict(label='cner', model=cner.JouleHeating(**dic)),
    ]
    plt.figure()
    t = np.linspace(0., 200., 4001)
    for d in mdl:
        plt.plot(t, d['model'].value(t), label=d['label'])
    plt.grid(True)
    plt.xlabel('Temperature (C)')
    plt.ylabel('Joule Heating per unit length (Wm**-1)')
    plt.legend()
    plt.show()

    return


def plot_solar_heating(dic):
    # olla not tested here since olla's solar heating is the same as ieee's one
    mdl = [
        dict(label='cigre', cm=cm.winter, model=cigre.SolarHeating(**dic)),
        dict(label='ieee', cm=cm.autumn, model=ieee.SolarHeating(**dic)),
        dict(label='cner', cm=cm.summer, model=cner.SolarHeating(**dic)),
    ]

    # examples 1
    month = np.array(range(1, 13))
    day = 21 * np.ones_like(month)
    hour = np.linspace(0, 24, 24 * 120 + 1)
    fig, ax = plt.subplots(nrows=1, ncols=len(mdl))
    for k in range(12):
        dic.update([('month', month[k]), ('day', day[k]), ('hour', hour)])
        for i, d in enumerate(mdl):
            d['model'].__init__(**dic)
            c = d['cm'](np.linspace(0., 1., len(day) + 2))[1:-1]
            ax[i].plot(hour, d['model'].value(0.), '-', c=c[k], label=calendar.month_name[k + 1])
            ax[i].set_title(d['label'])

    ax[0].set_ylabel('Solar Heating per unit length (Wm**-1)')
    for j in range(len(mdl)):
        ax[j].set_ylim([0, 25])
        ax[j].grid(True)
        ax[j].legend()
        ax[j].set_xlabel('Hour of day')

    # examples 2
    dr = pd.date_range(
        datetime.datetime(2001, 1, 1),
        datetime.datetime(2001, 12, 31),
        freq='h'
    )
    hl = np.linspace(0, 24, 13)[:-1]
    fig, ax = plt.subplots(nrows=1, ncols=len(mdl))
    for k, h in enumerate(hl):
        dic.update([('month', dr.month.values), ('day', dr.day.values), ('hour', h)])
        for i, d in enumerate(mdl):
            d['model'].__init__(**dic)
            c = d['cm'](np.linspace(0., 1., len(hl) + 2))[1:-1]
            ax[i].plot(dr, d['model'].value(0.), '-', c=c[k], label='At %02d:00' % (h,))
            ax[i].set_title(d['label'])
    ax[0].set_ylabel('Solar Heating per unit length (Wm**-1)')
    for j in range(len(mdl)):
        ax[j].set_ylim([0, 25])
        ax[j].grid(True)
        ax[j].legend()
        ax[j].set_xlabel('Day of year')

    plt.show()

    return


def plot_convective_cooling(dic):
    ws = np.linspace(0, 1, 5)
    wa = dic['azm'] - np.array([0, 45, 90])
    Tc = np.linspace(0., 80., 41)
    Ta = np.linspace(-10, 40, 6)

    # olla not tested here since olla's convective cooling is the same as ieee's one
    mdl = [
        dict(label='cigre', cm=cm.winter, model=cigre.ConvectiveCooling(**dic)),
        dict(label='ieee', cm=cm.autumn, model=ieee.ConvectiveCooling(**dic)),
        dict(label='cner', cm=cm.summer, model=cner.ConvectiveCooling(**dic)),
    ]

    fig, ax = plt.subplots(nrows=len(ws), ncols=len(wa))
    for i, u in enumerate(ws):
        for j, a in enumerate(wa):
            for m, ta in enumerate(Ta):
                dic.update([('ws', u), ('wa', a), ('Ta', ta)])
                for k, d in enumerate(mdl):
                    d['model'].__init__(**dic)
                    c = d['cm'](np.linspace(0., 1., len(Ta) + 2))[1:-1]
                    ax[i, j].plot(Tc, d['model'].value(Tc), '-', c=c[m], label='T$_a$=%.0f C' % (ta,))
                ax[i, j].set_title('At u=%.1f and $\phi$=%.0f' % (u, a))
                ax[i, j].grid(True)
                ax[i, j].set_ylim([-50, 100])
    for i in range(len(ws)):
        ax[i, 0].set_ylabel('Convective Cooling per unit length (Wm**-1)')
    for j in range(len(wa)):
        ax[-1, j].set_xlabel('Conductor temperature (C)')
    ax[0, -1].legend()

    plt.show()

    return


def plot_radiative_cooling(dic):
    Tc = np.linspace(0., 80., 41)
    Ta = np.linspace(-20, 50, 8)
    cl = cm.Spectral_r(np.linspace(0., 1., len(Ta) + 2)[1:-1])

    # only ieee and cner models are tested here since cigre, ieee and olla radiative coolings are all the same

    plt.figure()
    for i, ta in enumerate(Ta):
        dic['Ta'] = ta

        rc = ieee.RadiativeCooling(**dic)
        plt.plot(Tc, rc.value(Tc), c=cl[i], label='T$_a$=%.0f C' % (ta,))

        nc = cner.RadiativeCooling(**dic)
        plt.plot(Tc, nc.value(Tc), '--', c=cl[i])

    plt.grid(True)
    plt.title('NB: CNER model is the dotted line')
    plt.xlabel('Conductor temperature (C)')
    plt.ylabel('Radiative Cooling per unit length (Wm**-1)')
    plt.legend()

    plt.show()

    return


if __name__ == '__main__':
    """Check all power terms using default values. We use 45 deg for latitude 
    as the 0 in default does not show much variation for solar_heating."""

    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    dic = solver.default_values()
    dic['lat'] = 45.
    dic['tb'] = 0.
    dic['al'] = 0.25

    plot_joule_heating(dic)
    plot_solar_heating(dic)
    plot_convective_cooling(dic)
    plot_radiative_cooling(dic)
