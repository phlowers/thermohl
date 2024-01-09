#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import calendar
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from thermohl import sun


def _plot_seasons(ax=None, year=2001):
    """Add vertical line on graph to highlight season change."""

    if ax is None:
        ax = plt

    spr = datetime.datetime(year, 3, 22)
    smr = datetime.datetime(year, 6, 21)
    aut = datetime.datetime(year, 9, 20, 12)
    win = datetime.datetime(year, 12, 21)

    ax.axvline(x=spr, ymin=-23, ymax=+23, c='C0', ls='--', lw=1, label='spring')
    ax.axvline(x=smr, ymin=-23, ymax=+23, c='C1', ls='--', lw=1, label='summer')
    ax.axvline(x=aut, ymin=-23, ymax=+23, c='C2', ls='--', lw=1, label='autumn')
    ax.axvline(x=win, ymin=-23, ymax=+23, c='C3', ls='--', lw=1, label='winter')

    return


def test_hour_angle():
    """Plot hour angle over time (one day)."""

    dr = pd.date_range(datetime.datetime(2001, 4, 14, 20, 0, 0),
                       datetime.datetime(2001, 4, 16, 4, 0, 0),
                       freq='15min')
    plt.figure()
    ha = sun.hour_angle(dr.hour, minute=dr.minute)
    plt.plot(dr, ha / np.pi)
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Hour angle / $\pi$')

    plt.show()

    return


def test_solar_declination():
    """Plot solar declination over time (one year)."""

    dr = pd.date_range(datetime.datetime(2000, 12, 15),
                       datetime.datetime(2002, 1, 15),
                       freq='D')

    plt.figure()
    sd = np.rad2deg(sun.solar_declination(dr.month, dr.day))
    plt.plot(dr, sd, '-', c='k')
    _plot_seasons()
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Solar declination (deg)')

    plt.show()

    return


def test_solar_altitude_1(only_positive=False):
    """Plot solar altitude at different latitudes.

    See variation at fixed hours for different days of the year.

    """
    lt = np.linspace(0, 0.5 * np.pi, 7)[:-1]
    dr = pd.date_range(datetime.datetime(2001, 1, 1),
                       datetime.datetime(2001, 12, 31),
                       freq='D')
    hl = np.linspace(0, 24, 13)[:-1]
    cl = cm.magma(np.linspace(0., 1., len(hl) + 2))[1:-1]

    fig, ax = plt.subplots(nrows=2, ncols=len(lt))
    for j, lat in enumerate(lt):
        for i, sg in enumerate([-1, +1]):
            for k, h in enumerate(hl):
                sa = np.rad2deg(sun.solar_altitude(sg * lat, dr.month, dr.day, h))
                if only_positive:
                    sa = np.where(sa > 0., sa, 0.)
                sa = sa + np.where(h <= 12, 0, 1)
                ax[i, j].plot(dr, sa, c=cl[k], label='At %02d:00' % (h,))
            ax[i, j].grid(True)
            ax[i, j].set_title('At lat = %.1f' % (sg * np.rad2deg(lat),))
    for i in range(2):
        ax[i, 0].set_ylabel('Solar altitude (deg)')
    for j in range(len(lt)):
        ax[-1, j].set_xlabel('Day of year')
    ax[-1, -1].legend()

    plt.show()

    return


def test_solar_altitude_2(only_positive=False):
    """Plot solar altitude at different latitudes.

    See variation on the 21st of each month for different hours of the day.

    """
    lt = np.linspace(0, 0.5 * np.pi, 7)[:-1]
    mo = np.array(range(1, 13))
    dy = np.ones_like(mo) * 21
    hl = np.linspace(0, 24, 24 * 4)[:-1]
    cl = cm.Spectral(np.linspace(0., 1., len(mo) + 2))[1:-1]

    fig, ax = plt.subplots(nrows=2, ncols=len(lt))
    for j, lat in enumerate(lt):
        for i, sg in enumerate([-1, +1]):
            for k, m in enumerate(mo):
                sa = np.rad2deg(sun.solar_altitude(sg * lat, mo[k], dy[k], hl))
                if only_positive:
                    sa = np.where(sa > 0., sa, 0.)
                ax[i, j].plot(hl, sa, c=cl[k], label=calendar.month_name[k + 1])
            ax[i, j].grid(True)
            ax[i, j].set_title('At lat = %.1f' % (sg * np.rad2deg(lat),))
    for i in range(2):
        ax[i, 0].set_ylabel('Solar altitude (deg)')
    for j in range(len(lt)):
        ax[-1, j].set_xlabel('Hour of day')
    ax[-1, -1].legend()

    plt.show()

    return


def test_solar_azimuth_1(only_positive=False):
    """Plot solar azimuth at different latitudes.

    See variation at fixed hours for different days of the year.

    """
    lt = np.linspace(0, 0.5 * np.pi, 7)[:-1]
    dr = pd.date_range(datetime.datetime(2001, 1, 1),
                       datetime.datetime(2001, 12, 31),
                       freq='D')
    hl = np.linspace(0, 24, 13)[:-1]
    cl = cm.magma(np.linspace(0., 1., len(hl) + 2))[1:-1]

    fig, ax = plt.subplots(nrows=2, ncols=len(lt))
    for j, lat in enumerate(lt):
        for i, sg in enumerate([-1, +1]):
            for k, h in enumerate(hl):
                sa = np.rad2deg(sun.solar_azimuth(sg * lat, dr.month, dr.day, h))
                if only_positive:
                    sa = np.where(sun.solar_altitude(sg * lat, dr.month, dr.day, h) > 0., sa, 0.)
                sa = sa + np.where(h <= 12, 0, 1)
                ax[i, j].plot(dr, sa, c=cl[k], label='At %02d:00' % (h,))
            ax[i, j].grid(True)
            ax[i, j].set_title('At lat = %.1f' % (sg * np.rad2deg(lat),))
    for i in range(2):
        ax[i, 0].set_ylabel('Solar azimuth (deg)')
    for j in range(len(lt)):
        ax[-1, j].set_xlabel('Day of year')
    ax[-1, -1].legend()

    plt.show()

    return


def test_solar_azimuth_2(only_positive=False):
    """Plot solar azimuth at different latitudes.

    See variation on the 21st of each month for different hours of the day.

    """
    lt = np.linspace(0, 0.5 * np.pi, 7)[:-1]
    mo = np.array(range(1, 13))
    dy = np.ones_like(mo) * 21
    hl = np.linspace(0, 24, 24 * 4)[:-1]
    cl = cm.Spectral(np.linspace(0., 1., len(mo) + 2))[1:-1]

    fig, ax = plt.subplots(nrows=2, ncols=len(lt))
    for j, lat in enumerate(lt):
        for i, sg in enumerate([-1, +1]):
            for k, m in enumerate(mo):
                sa = np.rad2deg(sun.solar_azimuth(sg * lat, mo[k], dy[k], hl))
                if only_positive:
                    sa = np.where(sun.solar_altitude(sg * lat, mo[k], dy[k], hl) > 0., sa, 0.)
                ax[i, j].plot(hl, sa, c=cl[k], label=calendar.month_name[k + 1])
            ax[i, j].grid(True)
            ax[i, j].set_title('At lat = %.1f' % (sg * np.rad2deg(lat),))
    for i in range(2):
        ax[i, 0].set_ylabel('Solar azimuth (deg)')
    for j in range(len(lt)):
        ax[-1, j].set_xlabel('Hour of day')
    ax[-1, -1].legend()

    plt.show()

    return


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    test_hour_angle()
    test_solar_declination()
    test_solar_altitude_1()
    test_solar_altitude_2()
    test_solar_azimuth_1()
    test_solar_azimuth_2()
