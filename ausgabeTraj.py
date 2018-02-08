# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 20:25:12 2018

@author: Thorsten
"""
from lib import slam
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time



'''
Load data
'''
path = 'D:/KITTI/odometry/dataset/07_export/'
nrOfScans = 1100

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
fixResolution = 0.1

"""
Parameter particle Filter
"""
nrParticle = 1000
stddPos = 0.2
stddYaw = 0.025

trajs = []
nr = 50
for ii in range(0,nr):
    t0 = time.time()
    traj = slam.slatchNoResample(path,nrOfScans,bbPseudoRadius,
                           l_occupied,l_free,l_min,l_max,
                           fixResolution,
                           nrParticle,stddPos,stddYaw,'') 
    trajs.append(np.delete(traj,2,1))
    print('iteration = '+str(ii)+
          ' resolution =' +str(fixResolution)+
          ' time = ' + str(time.time()-t0) + 's')
          
trajs = np.hstack(trajs)
np.savetxt(path+'trajs.txt',trajs,delimiter=',')
