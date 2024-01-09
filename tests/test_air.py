# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from thermohl.air import Wikipedia, CIGRE, IEEE

matplotlib.use('TkAgg')
plt.close('all')

# temperature and altitude range
T = np.linspace(-20., 50., 701)
A = np.array([0., 500., 1000., 1500.])

# colors
C0 = cm.Blues_r(np.linspace(0., 1., len(A) + 4))[2:-2]
C1 = cm.Oranges_r(np.linspace(0., 1., len(A) + 4))[2:-2]
C2 = cm.Greens_r(np.linspace(0., 1., len(A) + 4))[2:-2]

fig, ax = plt.subplots(nrows=2, ncols=2)

# plot density
ax[0, 0].set_title('Volumic mass')
for k, a in enumerate(A):
    ax[0, 0].plot(T, Wikipedia.volumic_mass(T, a), c=C0[k], label='Wikipedia at %.0f m' % (a,))
for k, a in enumerate(A):
    ax[0, 0].plot(T, CIGRE.volumic_mass(T, a), c=C1[k], label='CIGRE at %.0f m' % (a,))
for k, a in enumerate(A):
    ax[0, 0].plot(T, IEEE.volumic_mass(T, a), c=C2[k], label='IEEE at %.0f m' % (a,))

# plot viscosity
ax[0, 1].set_title('Dynamic viscosity')
ax[0, 1].plot(T, Wikipedia.dynamic_viscosity(T), c='C0', lw=2., label='Wikipedia')
for k, a in enumerate(A):
    ax[0, 1].plot(T, CIGRE.dynamic_viscosity(T, a), c=C1[k], label='CIGRE at %.0f m' % (a,))
ax[0, 1].plot(T, IEEE.dynamic_viscosity(T), c='C2', label='IEEE')

# plot conductivity
ax[1, 0].set_title('Thermal conductivity')
ax[1, 0].plot(T, Wikipedia.thermal_conductivity(T), c='C0', lw=2., label='Wikipedia')
ax[1, 0].plot(T, CIGRE.thermal_conductivity(T), c='C1', label='CIGRE')
ax[1, 0].plot(T, IEEE.thermal_conductivity(T), c='C2', label='IEEE')

# plot prandtl
ax[1, 1].set_title('Prandtl number')
ax[1, 1].plot(T, CIGRE.prandtl(T), c='C1', label='CIGRE')

# grid, legend and axe labeling
for i in range(2):
    for j in range(2):
        ax[i, j].grid(True)
        ax[i, j].legend()
ax[1, 0].set_xlabel('Temperature (C)')
ax[1, 1].set_xlabel('Temperature (C)')

plt.show()
