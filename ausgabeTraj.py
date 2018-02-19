# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 20:25:12 2018

@author: Thorsten
"""
from lib import slam
import numpy as np
import time



'''
Load data
'''
path = 'C:/KITTI/2011_09_26/2011_09_26_drive_0117_export/'
nrOfScans = 659

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
Parameter particle Filter
"""
stddPos = 0.2
stddYaw = 0.025

"""
Parameter grid
"""
fixResolution = 0.05

for ii in range(0,10):
     
    
    trajs = []
    
    nr = 25
    for ii in range(0,nr):
        t0 = time.time()
        traj = slam.slatch2Resample(path,nrOfScans,bbPseudoRadius,
                               l_occupied,l_free,l_min,l_max,
                               fixResolution,
                               stddPos,stddYaw,'') 
        trajs.append(np.delete(traj,2,1))
        print('iteration = '+str(ii)+
              ' resolution =' +str(fixResolution)+
              ' time = ' + str(time.time()-t0) + 's')
              
    trajs = np.hstack(trajs)
    np.savetxt(path+'trajs_'+str(fixResolution)+'.txt',trajs,delimiter=',')
    
    fixResolution = fixResolution + 0.05 
