# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:29:34 2018

@author: Thorsten
"""

import numpy as np
import matplotlib.pyplot as plt

font = {'size'   : 20}
plt.rc('font', **font)

length = 100
resolution = 0.25

grid = np.zeros((int(length/resolution),int(length/resolution)),order='C')
print(grid.nbytes/1000/1000/1000)

maxSize = 0.5*10**9

maxLength = np.sqrt(maxSize/8*resolution**2)

print(maxLength)


resolution = 0.10
length = np.linspace(0,10000,1000)
size = (length/resolution)**2*8/10**6
plt.plot(length/1000,size/1000,label='Zellgröße = 0.10m')

resolution = 0.25
length = np.linspace(0,10000,1000)
size = (length/resolution)**2*8/10**6
plt.plot(length/1000,size/1000,label='Zellgröße = 0.25m')

resolution = 0.40
length = np.linspace(0,10000,1000)
size = (length/resolution)**2*8/10**6
plt.plot(length/1000,size/1000,label='Zellgröße = 0.40m')


plt.xlabel('Kantenlänge quadratische Karte [km]')
plt.ylabel('Speicherbedarf [GB]')
plt.title('Speicherbedarf Occupancy Grid')
plt.legend(loc='lower right')
plt.grid()
plt.ylim((0,4))