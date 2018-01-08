# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:08:15 2018

@author: Thorsten
"""

import numpy as np
import matplotlib.pyplot as plt
import math

basedir = 'C:/Users/Thorsten/ownCloud/Shared/Studienarbeit/Programmierung/MappingWithKnownPoses/eva_20110926_0117/'

plt.figure(1)
plt.subplot(212)
plt.title('Abweichung Position zu Ground Truth')
plt.xlabel('Distanz GT [m]')
plt.ylabel('Abweichung [m]')
plt.subplot(211)
plt.title('Abweichung Yaw zu Ground Truth')
plt.xlabel('Distanz GT [m]')
plt.ylabel('Abweichung [°]')


evaFile = np.loadtxt(open(basedir+'eva_2011_09_26_0117_10000p_30f_02_0025_01.txt'),delimiter=',')
distGT = evaFile[:,0]

"""
10000 particle
"""
plt.subplot(212)
plt.plot(distGT,evaFile[:,1],color='g')
plt.subplot(211)
plt.plot(distGT,evaFile[:,2]/math.pi*180,color='g')

evaFile = np.loadtxt(open(basedir+'eva_2011_09_26_0117_10000p_30f_02_0025_02.txt'),delimiter=',')
plt.subplot(212)
plt.plot(distGT,evaFile[:,1],color='g')
plt.subplot(211)
plt.plot(distGT,evaFile[:,2]/math.pi*180,color='g')


"""
1000 particle
"""
evaFile = np.loadtxt(open(basedir+'eva_2011_09_26_0117_1000p_30f_02_0025_01.txt'),delimiter=',')
plt.subplot(212)
plt.plot(distGT,evaFile[:,1],color='y')
plt.subplot(211)
plt.plot(distGT,evaFile[:,2]/math.pi*180,color='y')

evaFile = np.loadtxt(open(basedir+'eva_2011_09_26_0117_1000p_30f_02_0025_02.txt'),delimiter=',')
plt.subplot(212)
plt.plot(distGT,evaFile[:,1],color='y')
plt.subplot(211)
plt.plot(distGT,evaFile[:,2]/math.pi*180,color='y')

evaFile = np.loadtxt(open(basedir+'eva_2011_09_26_0117_1000p_30f_02_0025_03.txt'),delimiter=',')
plt.subplot(212)
plt.plot(distGT,evaFile[:,1],color='y')
plt.subplot(211)
plt.plot(distGT,evaFile[:,2]/math.pi*180,color='y')

evaFile = np.loadtxt(open(basedir+'eva_2011_09_26_0117_1000p_30f_02_0025_04.txt'),delimiter=',')
plt.subplot(212)
plt.plot(distGT,evaFile[:,1],color='y')
plt.subplot(211)
plt.plot(distGT,evaFile[:,2]/math.pi*180,color='y')

evaFile = np.loadtxt(open(basedir+'eva_2011_09_26_0117_1000p_30f_02_0025_05.txt'),delimiter=',')
plt.subplot(212)
plt.plot(distGT,evaFile[:,1],color='y')
plt.subplot(211)
plt.plot(distGT,evaFile[:,2]/math.pi*180,color='y')

evaFile = np.loadtxt(open(basedir+'eva_2011_09_26_0117_1000p_30f_02_0025_06.txt'),delimiter=',')
plt.subplot(212)
plt.plot(distGT,evaFile[:,1],color='y')
plt.subplot(211)
plt.plot(distGT,evaFile[:,2]/math.pi*180,color='y')

"""
100 particle
"""
evaFile = np.loadtxt(open(basedir+'eva_2011_09_26_0117_100p_30f_02_0025_01.txt'),delimiter=',')
plt.subplot(212)
plt.plot(distGT,evaFile[:,1],color='r')
plt.subplot(211)
plt.plot(distGT,evaFile[:,2]/math.pi*180,color='r')

evaFile = np.loadtxt(open(basedir+'eva_2011_09_26_0117_100p_30f_02_0025_02.txt'),delimiter=',')
plt.subplot(212)
plt.plot(distGT,evaFile[:,1],color='r')
plt.subplot(211)
plt.plot(distGT,evaFile[:,2]/math.pi*180,color='r')

evaFile = np.loadtxt(open(basedir+'eva_2011_09_26_0117_100p_30f_02_0025_03.txt'),delimiter=',')
plt.subplot(212)
plt.plot(distGT,evaFile[:,1],color='r')
plt.subplot(211)
plt.plot(distGT,evaFile[:,2]/math.pi*180,color='r')