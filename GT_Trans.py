# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:21:05 2018

@author: Thorsten
"""

import numpy as np
import matplotlib.pyplot as plt

path = 'C:/KITTI/2011_09_30/2011_09_30_0027_export/'

groundTruth = np.asmatrix(np.loadtxt(path+'groundTruth.txt',delimiter=','))
calib = np.asmatrix(np.loadtxt(path+'calib.txt',delimiter=','))

T = calib[0:2,3]

groundTruthTrans = []
for ii in range(0,np.shape(groundTruth)[0]):
    yaw = groundTruth[ii,2]
    R = np.matrix([[np.cos(yaw), -np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    groundTruthTrans.append(-np.transpose(R*T)+groundTruth[ii,0:2])
groundTruthTrans = np.vstack(groundTruthTrans)
    

plt.figure(0)
plt.scatter([groundTruth[::10,0]],
            [groundTruth[::10,1]],
            c='b',s=20,edgecolors='none', label = 'Trajektorie GT IMU')
plt.scatter([groundTruthTrans[::10,0]],
            [groundTruthTrans[::10,1]],
            c='m',s=20,edgecolors='none', label = 'Trajektorie GT Velo')
plt.legend()
plt.figure(1)        
plt.plot(groundTruth[:,2])