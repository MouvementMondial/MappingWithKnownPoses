# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:52:48 2017

@author: Thorsten
"""

import numpy as np
import pykitti
import utm
import time

from lib import mapping
from lib import transform

basedir = 'C:/KITTI'
date = '2011_09_30'
drive = '0027'

dataset = pykitti.raw(basedir, date, drive, imformat='cv2')
it = dataset.oxts

"""
Parameter PCL
"""
# The IMU is mounted at 0.63m height
zCutMin = -0.05
zCutMax = 0.05

"""
Parameter mapping from Octomap paper
"""
l_occupied = 0.85
l_free = -0.4
l_min = -2.0
l_max = 3.5


"""
Create empty GRID [m] and initialize it with 0, this Log-Odd 
value is maximum uncertainty (P = 0.5)
"""
length = 1000.0
width = 1000.0
resolution = 0.1

grid = np.zeros((int(length/resolution),
                 int(width/resolution)),
                 order='C')

offset = np.array([[0,0]])                 

"""
Extract Poinctlouds from datastructure, transform them to world coordinate system

Add this measurement to 
"""
traj = []
nr = 0
for scan in dataset.velo:
    t0 = time.time()
    print('Process measurement: ' + str(nr))
    
    # get position and orientation angles from trajectory
    pose = next(it)
    lat = pose.packet.lat
    lon = pose.packet.lon
    alt = pose.packet.alt
    roll = pose.packet.roll 
    pitch = pose.packet.pitch
    yaw = pose.packet.yaw
    
    # LaLo to UTM
    pos_cartesian = utm.from_latlon(lat,lon)
    
    # create rotation and translation matrices
    R = transform.rotationMatrix_ypr(yaw,pitch,roll)
    T = np.matrix([pos_cartesian[0], pos_cartesian[1], alt])

    pos_sensor = np.array([[pos_cartesian[0], pos_cartesian[1]]])
    
    # first measurement: get offset
    if nr == 0:
        offset = np.array([[pos_sensor[0,0]-length/2,
                            pos_sensor[0,1]-width/2]])
        
    # get PCL and delete intensity
    points = np.asarray(scan)
    points = np.delete(points,3,1)

    # transform points
    points = transform.transformPointcloud(points, R, T)
    
    # get pcl where z-value is in a certain range
    # schneller: vorauswahl der Höhe bereits vor der Transformation und dann 
    # nochmal Feinfilterung der Höhe
    zCutMin_global = zCutMin + alt
    zCutMax_global = zCutMax + alt

    binary = np.logical_and( points[:,2]>zCutMin_global, points[:,2]<zCutMax_global)  
    binary3 = np.column_stack((binary,binary,binary)) 
    pointsFiltered = points[binary3]
    pointsFiltered = np.reshape(pointsFiltered,(-1,3))
    
    mapping.addMeasurement(grid,pointsFiltered[:,0],pointsFiltered[:,1],pos_sensor,offset,resolution,l_occupied,l_free,l_min,l_max)
  
    # save traj point
    traj.append(T)
    nr = nr + 1
    dt = time.time() - t0
    print('Duration: %.2f' % dt)
  
# save map
plt.imsave('grid_'+str(n)+'.png', grid[:,:], cmap = 'binary') 