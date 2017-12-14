# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:52:48 2017

@author: Thorsten
"""

import numpy as np
import matplotlib.pyplot as plt
import pykitti
import utm
import time

from lib import mapping
from lib import transform
from lib import io
from lib import filterPCL


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

bbPseudoRadius = 30.0

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
length = 100.0
width = 100.0
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

    """
    Read measurement from KITTI-Datastructure
    """    
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
    
    pos_sensor = np.array([[pos_cartesian[0], pos_cartesian[1]]])
    
    # first measurement: get offset
    if nr == 0:
        offset = np.array([[pos_sensor[0,0]-length/2,
                            pos_sensor[0,1]-width/2]])

    """
    Filter pointcloud and rotate with pitch and roll from pos-system to street plane
    """      
    # get PCL and delete intensity
    points = np.asarray(scan)
    points = np.delete(points,3,1)
    
    # limit range 
    points = filterPCL.filterBB(points,bbPseudoRadius,0.0,0.0)
    
    # get pcl where z-value is in a certain range, in street plane!
    points = filterPCL.filterZ(points,zCutMin,zCutMax)
    
    # rotate points to street plane -> pitch and roll
    points = transform.rotatePointcloud(points,transform.rotationMatrix_ypr(0,pitch,roll))       
    
    io.writePcl2xxy(points,'pcl_'+str(nr)+'.txt')
    
    # project points to xy-plane (delete z values)
    points = np.delete(points,2,1)
    
    print(nr)

    print(yaw)    
    
    """
    Estimate Position with pos-system and transform pointcloud to 
    """
    # rotate points with yaw
    points = transform.rotatePointcloud(points, transform.rotationMatrix_2D(yaw))
    
    # translate points to global coord system
    T = np.matrix([pos_cartesian[0], pos_cartesian[1]])
    points = transform.translatePointcloud(points,T)
    
    # calculate distance from grid
    distance = mapping.scan2mapDistance(grid,points,offset,resolution)
      
    print('Distance: %.2f' % distance)

    """
    Add measurement to grid and save trajectory point
    """
    # add measurement to grid    
    mapping.addMeasurement(grid,points[:,0],points[:,1],pos_sensor,offset,resolution,l_occupied,l_free,l_min,l_max)
    # save traj point
    traj.append(T)
    nr = nr + 1
    dt = time.time() - t0
    print('Duration: %.2f' % dt)
    
    io.writeGrid2file(grid,'grid_'+str(nr)+'.txt')
    
# save map
io.writeGrid2Img(grid,'grid_'+str(nr)+'.png')
io.writeGrid2Pcl(grid,'gridALSpcl.txt',offset,resolution,l_max,l_min)