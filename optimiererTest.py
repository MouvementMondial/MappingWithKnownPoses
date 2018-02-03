# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:30:14 2018

@author: Thorsten
"""

import numpy as np
from numba import jit




Nfeval = 1

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

@jit(nopython=True)
def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x-x**2.0)**2.0 + (1-x)**2.0)
    

x0 = np.array([150])
    
#res = minimize(rosen, x0, method='Powell',
#               options={'xtol': 1e-8, 'disp': True})
               
#res2 = minimize(rosen,x0,method='L-BFGS-B',options={'disp': True},bounds=((0,160),))

res3 = differential_evolution(rosen,((0,160),))