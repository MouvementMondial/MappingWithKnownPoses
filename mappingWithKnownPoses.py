# -*- coding: utf-8 -*-
"""
@author: Thorsten

Thanks to balzer82
https://github.com/balzer82/3D-OccupancyGrid-Python
"""

import numpy as np
import matplotlib.pyplot as plt

from lib import mapping
from lib import io
import time

"""
Parameter
"""
folder = 'C:/Users/Thorsten/ownCloud/Shared/Studienarbeit/Programmierung/Testdaten/20110930_027'

"""
Load Trajectory (known Poses)
"""
traj = io.readTraj2xyz(folder+'/traj.txt')
nr_of_scans = traj.shape[0]

"""
Create List with filenames of the Pointclouds
"""
filenames_pcl = []

for m in range(nr_of_scans):
    filenames_pcl.append(folder+'/pcl_filter/pointcloud_zFiltered_'+str(m)+'.txt')

"""
Create empty GRID [m] and initialize it with 0, this Log-Odd 
value is maximum uncertainty (P = 0.5)
"""
length = 1000.0
width = 1000.0

resolution = 0.1

#grid = np.zeros((int(length/resolution),
#                 int(width/resolution)),
#                 dtype=np.float16)

grid = np.zeros((int(length/resolution),
                 int(width/resolution)),
                 order='C')


"""
Offset from first pose
"""
pos_sensor = np.array([[traj[0,0],traj[0,1]]])

offset = np.array([[pos_sensor[0,0]-length/2,
                    pos_sensor[0,1]-width/2]])

"""
Parameter from Octomap Paper
"""
l_occupied = 0.85
l_free = -0.4

l_min = -2.0
l_max = 3.5

"""
Mapping for n poses
"""
for n in range(nr_of_scans):
    
    print('Processing scan nr: '+str(n))
    """
    Load Measurement
    """
    pcl = io.readPointcloud2xyz(filenames_pcl[n])
    x = pcl[:,[0]]
    y = pcl[:,[1]]
    pos_sensor = np.array([[traj[n,0],traj[n,1]]])

    """
    Add measurement to GRID
    """
    t0 = time.time()
    mapping.addMeasurement(grid,x,y,pos_sensor,offset,resolution,l_occupied,l_free,l_min,l_max)   
                    
    """
    Show or save GRID
    """
    #plt.figure()
    #plt.imshow(grid[:,:], interpolation ='none', cmap = 'binary')
    
    #time.sleep(0.2) # avoid bug while writing image
    #plt.imsave('grid_'+str(n)+'.png', grid[:,:], cmap = 'binary')  

    dt = time.time() - t0
    print('Duration: %.2f' % dt)   

plt.imsave('grid_'+str(n)+'.png', grid[:,:], cmap = 'binary')      