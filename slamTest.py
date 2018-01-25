# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:52:37 2018

@author: Thorsten
"""

from lib import slam

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
resolution = 0.1

"""
Parameter particle Filter
"""
nrParticle = 1000
stddPos = 0.2
stddYaw = 0.025

UID = 'Test'
errorSums = []
for ii in range(0,10):
    errorSum = slam.slatch(path,nrOfScans,bbPseudoRadius,
                           l_occupied,l_free,l_min,l_max,
                           resolution,
                           nrParticle,stddPos,stddYaw,UID+str(ii)) 
    errorSums.append(errorSum)
    print('errorSum = '+str(errorSum))
    
print(errorSums)