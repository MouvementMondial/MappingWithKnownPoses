# -*- coding: utf-8 -*-
"""
@author: Thorsten
"""

from lib import bresenham
import numpy as np
from numba import jit

@jit
def addMeasurement(grid, x, y, pos_sensor, offset, resolution, l_occupied, l_free, l_min, l_max):
    
    for i in range(x.size):
        # round points to cells 
        xi=int( (x[i,0]-offset[0,0]) / resolution )
        yi=int( (y[i,0]-offset[0,1]) / resolution )

        # set beam endpoint-cells as occupied
        grid[xi,yi] += l_occupied
    
        # value > threshold? -> clamping 
        if grid[xi,yi] > l_max:
            grid[xi,yi] = l_max

        # calculate cells between sensor and endpoint as free
        path = bresenham.bresenham2D( ((pos_sensor-offset)/resolution).astype(int), np.array([[xi,yi]]))
        
        # set cells between sensor and endpoint as free
        updateFree(path,grid,l_free,l_min)
            
@jit(nopython=True)
def updateFree(path,grid,l_free,l_min):
    for nr in range(path.shape[0]):
        path_x = int(path[nr,0])
        path_y = int(path[nr,1])
        
        grid[path_x, path_y] += l_free
        
        # value < threshold? -> clamping
        if grid[path_x, path_y] < l_min:
            grid[path_x, path_y] = l_min
            
@jit(nopython=True)
def scan2mapDistance(grid,pcl,offset,resolution):
    distance = 0;
    for i in range(pcl.shape[0]):
        # round points to cells
        xi = int ( (pcl[i,0]-offset[0,0]) / resolution )
        yi = int ( (pcl[i,1]-offset[0,1]) / resolution )
        if grid[xi,yi] > 0:    
            distance += grid[xi,yi]
    return distance