# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:52:37 2018

@author: Thorsten
"""

from lib import slam
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time

def slatchOpt(resolution):
    
    '''
    Load data
    '''
    path = 'D:/KITTI/odometry/dataset/04_export/'
    nrOfScans = 270
    
    """
    Parameter PCL
    """
    bbPseudoRadius = 30.0
    
    """
    Parameter mapping from Octomap paper
    """
    l_occupied = 0.85
    l_free = -0.4
    l_min = -2.0
    l_max = 3.5
    
    """
    Parameter grid
    """
    fixResolution = resolution[0]
    
    """
    Parameter particle Filter
    """
    nrParticle = 1000
    stddPos = 0.2
    stddYaw = 0.025


    errorSums = []
    nr = 10
    for ii in range(0,nr):
        t0 = time.time()
        errorSum = slam.slatchNoResample(path,nrOfScans,bbPseudoRadius,
                               l_occupied,l_free,l_min,l_max,
                               fixResolution,
                               nrParticle,stddPos,stddYaw,'') 
        errorSums.append(errorSum)
        print('iteration = '+str(ii)+
              ' resolution =' +str(fixResolution)+
              ' errorSum = '+str(errorSum)+
              ' time = ' + str(time.time()-t0) + 's')
    print('errorSumAvg:'+str(np.vstack(errorSums).sum()/nr))
    return np.vstack(errorSums).sum()/nr


resolution = 0.2
#res = minimize(slatchOpt,resolution,method='TNC',options={'disp': True,'xtol': 0.01},bounds=((0.05,0.5),))
res = differential_evolution(slatchOpt,((0.05,0.5),))
