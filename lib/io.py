# -*- coding: utf-8 -*-
"""
@author: Thorsten
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readPointcloud2xyz(filename):
    pcl = pd.read_csv(filename, delimiter=',')
    return pcl.as_matrix()
        
def readTraj2xyz(filename):
    traj = pd.read_csv(filename, delimiter=',')
    return traj.as_matrix()
    
def writeGrid2Pcl(grid,filename,offset,resolution, l_max, l_min):
    file = open(filename,'w')    
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if(grid[x,y] >= l_max-2):
                xpcl = x*resolution+offset[0,0]
                ypcl = y*resolution+offset[0,1]
                file.write(str(xpcl)+' '+str(ypcl)+'\n')
    file.close

def writeGrid2Img(grid,filename):
    plt.imsave(filename,grid[:,:],cmap='binary')

def writePcl2xxy(pcl,filename):
    np.savetxt(filename,pcl,delimiter=',',fmt='%1.7f')