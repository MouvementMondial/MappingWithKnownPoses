# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:07:28 2017

@author: Thorsten
"""

import numpy as np
from numba import jit

@jit
def filterZ(pcl,zMin,zMax):
    binary = np.logical_and( pcl[:,2]>zMin, pcl[:,2]<zMax)  
    binary = np.column_stack((binary,binary,binary)) 
    pclFiltered = pcl[binary]
    return np.reshape(pclFiltered,(-1,3))

@jit
def filterBB(pcl,pseudoRadius,centerX,centerY):
    xMin = centerX-pseudoRadius
    xMax = centerX+pseudoRadius
    yMin = centerY-pseudoRadius
    yMax = centerY+pseudoRadius
    binary = np.logical_and( 
             np.logical_and( pcl[:,0]<xMax, pcl[:,0]>xMin ) ,
             np.logical_and( pcl[:,1]<yMax, pcl[:,1]>yMin ) )
    binary = np.column_stack((binary,binary,binary))
    pclFiltered = pcl[binary]
    return np.reshape(pclFiltered,(-1,3))