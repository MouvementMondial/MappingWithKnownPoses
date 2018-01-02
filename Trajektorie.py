# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:52:48 2017

@author: Thorsten
"""

import numpy as np
import matplotlib.pyplot as plt
import pykitti
import utm
import math

from lib import mapping
from lib import transform
from lib import io
from lib import filterPCL


basedir = 'C:/KITTI'
date = '2011_09_30'
drive = '0027'

dataset = pykitti.raw(basedir, date, drive, imformat='cv2')
it = dataset.oxts

delta_t = 0.1

"""
Extract Poinctlouds from datastructure, transform them to world coordinate system

Add this measurement to 
"""
trajGPS = []
trajVH = []
trajVW = []
trajAW = []

nr = 0
for scan in dataset.velo:
    pose = next(it)
    if nr == 0:
        # first measurement: get start position (offset)
        lat = pose.packet.lat
        lon = pose.packet.lon             
        # LaLo to UTM
        pos_cartesian = utm.from_latlon(lat,lon)    
        T = np.matrix([pos_cartesian[0], 
                       pos_cartesian[1], 
                       pose.packet.yaw,
                       pose.packet.vf,
                       pose.packet.af,
                       pose.packet.wu])

        trajGPS.append(T)
        trajVH.append(T)
        trajVW.append(T)
        trajAW.append(T)
    
    if nr != 0:
        # Calculate Position with GPS
        lat = pose.packet.lat
        lon = pose.packet.lon  
        pos_cartesian = utm.from_latlon(lat,lon)    
        T = np.matrix([pos_cartesian[0], 
                       pos_cartesian[1], 
                       pose.packet.yaw,
                       pose.packet.vf,
                       pose.packet.af,
                       pose.packet.wu])
        trajGPS.append(T)
    
        # Calculate Position with VH
        # Ungenauigkeit aus velacc von OXTS
        #vf = np.random.normal(pose.packet.vf,0.012)
        # 2 Grad (Genauigkeit Billigkompass für py)
        #yaw = np.random.normal(pose.packet.yaw,2.0/180.0*3.1415)
        oldPos = trajVH[-1]
        yaw = oldPos[0,2]        
        vf = oldPos[0,3]       
        T = np.matrix([oldPos[0,0]+math.cos(yaw)*vf*delta_t,
                       oldPos[0,1]+math.sin(yaw)*vf*delta_t,
                       pose.packet.yaw,
                       pose.packet.vf,
                       pose.packet.af,
                       pose.packet.wu])
        trajVH.append(T)      

        # Calculate Position with VW
        oldPos = trajVW[-1]
        vf = oldPos[0,3]
        wu = oldPos[0,5]
        yaw = oldPos[0,2]+delta_t*wu 
        
        T = np.matrix([oldPos[0,0]+math.cos(yaw)*vf*delta_t,
                       oldPos[0,1]+math.sin(yaw)*vf*delta_t,
                       yaw,
                       pose.packet.vf,
                       pose.packet.af,
                       pose.packet.wu])              
        trajVW.append(T)
        
        # Calculate Position with AW
        oldPos = trajAW[-1]
        af = oldPos[0,4]
        vf = oldPos[0,3]+af*delta_t
        wu = oldPos[0,5]
        yaw = oldPos[0,2]+delta_t*wu 
        T = np.matrix([oldPos[0,0]+math.cos(yaw)*vf*delta_t,
                       oldPos[0,1]+math.sin(yaw)*vf*delta_t,
                       yaw,
                       vf,
                       pose.packet.af,
                       pose.packet.wu])
        trajAW.append(T)
                       
    nr += 1
    


trajGPS = np.vstack(trajGPS)
np.savetxt('trajGPS.txt',trajGPS,delimiter=',',fmt='%1.3f')

trajVH = np.vstack(trajVH)  
np.savetxt('trajVH.txt',trajVH,delimiter=',',fmt='%1.3f')

trajVW = np.vstack(trajVW)  
np.savetxt('trajVW.txt',trajVW,delimiter=',',fmt='%1.3f')

trajAW = np.vstack(trajAW)  
np.savetxt('trajAW.txt',trajAW,delimiter=',',fmt='%1.3f')

gps = plt.scatter([trajGPS[:,0]],[trajGPS[:,1]],c='g',s=5,edgecolors='none')
vh = plt.scatter([trajVH[:,0]],[trajVH[:,1]],c='m',s=5,edgecolors='none')
vw = plt.scatter([trajVW[:,0]],[trajVW[:,1]],c='y',s=5,edgecolors='none')
aw = plt.scatter([trajAW[:,0]],[trajAW[:,1]],c='b',s=5,edgecolors='none')
plt.title('Varianten Trajektorienberechnung')
plt.xlabel('Rechtswert')
plt.ylabel('Hochwert')
plt.legend((gps,vh,vw,aw),
           ('Koordinaten','v_f und yaw','v_f und w_u','a_f und w_u'))