# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:30:14 2018

@author: Thorsten
"""

import numpy as np
from scipy.optimize import minimize

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x-x**2.0)**2.0 + (1-x)**2.0)
    
x0 = np.array([150])
    
res = minimize(rosen, x0, method='Powell',
               options={'xtol': 1e-8, 'disp': True})