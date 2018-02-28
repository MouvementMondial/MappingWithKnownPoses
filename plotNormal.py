# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:18:24 2018

@author: Thorsten
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

font = {'size'   : 20}
plt.rc('font', **font)

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x,mlab.normpdf(x, mu, sigma))

plt.grid()
plt.show()

