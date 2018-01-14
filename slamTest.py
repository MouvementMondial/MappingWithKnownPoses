# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:52:37 2018

@author: Thorsten
"""

import numpy as np
import pykitti
import math
import matplotlib.pyplot as plt

from lib import slam

'''
Load data
'''
# raw
basedir = 'C:/KITTI'
date = '2011_09_26'
drive = '0117'
dataset = pykitti.raw(basedir, date, drive, imformat='cv2')

"""
Parameter PCL
"""
# The IMU is mounted at 0.63m height
zCutMin = -0.05
zCutMax = 0.05

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
length = 500.0
width = 500.0
resolution = 0.25
"""
Parameter particle Filter
"""
nrParticle = 1000
stddPos = 0.2
stddYaw = 0.025

plt.figure(1)
plt.subplot(211)
plt.title('Abweichung Position zu Ground Truth')
plt.xlabel('Distanz GT [m]')
plt.ylabel('Abweichung [m]')

plt.subplot(212)
plt.title('Abweichung Yaw zu Ground Truth')
plt.xlabel('Distanz GT [m]')
plt.ylabel('Abweichung [°]')

nor = 10

nrParticle = 1000
bbPseudoRadius = 30
evaluation = slam.slatch(dataset,zCutMin,zCutMax,bbPseudoRadius,
                             l_occupied,l_free,l_min,l_max,
                             length,width,resolution,
                             nrParticle,stddPos,stddYaw)      
plt.subplot(211)
plt.plot(evaluation[:,0],evaluation[:,1],color = 'g',label = 'sample')
plt.subplot(212)
errYaw = np.asarray(evaluation[:,2]/math.pi*180)
errYaw[errYaw>=180.0] -= 360.0
errYaw[errYaw<=-180.0] += 360.0
plt.plot(evaluation[:,0],errYaw,color = 'g',label = 'sample')  
for _ in range(nor-1):
    evaluation = slam.slatch(dataset,zCutMin,zCutMax,bbPseudoRadius,
                             l_occupied,l_free,l_min,l_max,
                             length,width,resolution,
                             nrParticle,stddPos,stddYaw)      
    plt.subplot(211)
    plt.plot(evaluation[:,0],evaluation[:,1],color = 'g')
    plt.subplot(212)
    errYaw = np.asarray(evaluation[:,2]/math.pi*180)
    errYaw[errYaw>=180.0] -= 360.0
    errYaw[errYaw<=-180.0] += 360.0
    plt.plot(evaluation[:,0],errYaw,color = 'g') 

nrParticle = 1000   
bbPseudoRadius = 30 
evaluation = slam.slatchRE(dataset,zCutMin,zCutMax,bbPseudoRadius,
                             l_occupied,l_free,l_min,l_max,
                             length,width,resolution,
                             nrParticle,stddPos,stddYaw)      
plt.subplot(211)
plt.plot(evaluation[:,0],evaluation[:,1],color = 'm',label = 'resample')
plt.subplot(212)
errYaw = np.asarray(evaluation[:,2]/math.pi*180)
errYaw[errYaw>=180.0] -= 360.0
errYaw[errYaw<=-180.0] += 360.0
plt.plot(evaluation[:,0],errYaw,color = 'm',label = 'resample')  
for _ in range(nor-1):
    evaluation = slam.slatchRE(dataset,zCutMin,zCutMax,bbPseudoRadius,
                             l_occupied,l_free,l_min,l_max,
                             length,width,resolution,
                             nrParticle,stddPos,stddYaw)      
    plt.subplot(211)
    plt.plot(evaluation[:,0],evaluation[:,1],color = 'm')
    plt.subplot(212)
    errYaw = np.asarray(evaluation[:,2]/math.pi*180)
    errYaw[errYaw>=180.0] -= 360.0
    errYaw[errYaw<=-180.0] += 360.0
    plt.plot(evaluation[:,0],errYaw,color = 'm') 

"""
nrParticle = 1000  
bbPseudoRadius = 60
evaluation = slam.slatch(dataset,zCutMin,zCutMax,bbPseudoRadius,
                             l_occupied,l_free,l_min,l_max,
                             length,width,resolution,
                             nrParticle,stddPos,stddYaw)      
plt.subplot(211)
plt.plot(evaluation[:,0],evaluation[:,1],color = 'r',label = '60m')
plt.subplot(212)
errYaw = np.asarray(evaluation[:,2]/math.pi*180)
errYaw[errYaw>=180.0] -= 360.0
errYaw[errYaw<=-180.0] += 360.0
plt.plot(evaluation[:,0],errYaw,color = 'r',label = '60m')  
for _ in range(nor-1):
    evaluation = slam.slatch(dataset,zCutMin,zCutMax,bbPseudoRadius,
                             l_occupied,l_free,l_min,l_max,
                             length,width,resolution,
                             nrParticle,stddPos,stddYaw)      
    plt.subplot(211)
    plt.plot(evaluation[:,0],evaluation[:,1],color = 'r')
    plt.subplot(212)
    errYaw = np.asarray(evaluation[:,2]/math.pi*180)
    errYaw[errYaw>=180.0] -= 360.0
    errYaw[errYaw<=-180.0] += 360.0
    plt.plot(evaluation[:,0],errYaw,color = 'r')
"""
plt.subplot(211)
plt.legend(loc='upper left')
plt.subplot(212)
plt.legend(loc='upper left')