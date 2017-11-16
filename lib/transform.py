import numpy as np    

# -*- coding: utf-8 -*-
"""

Create rotation matrices, apply transformations to points

@author: Thorsten
"""

def rotationMatrix_ypr(yaw, pitch, roll):
    """
    Create rotation matrix for 3 rotations according to DIN8855 to rotate points
    from a global coordinate system to vehicle coordinate system
    
    Thanks to balzer82 (github)
    http://www.cbcity.de/tutorial-rotationsmatrix-und-quaternion-einfach-erklaert-in-din70000-zyx-konvention
    
    """
    
    Rr = np.matrix([[1.0, 0.0, 0.0],[0.0, np.cos(roll), -np.sin(roll)],[0.0, np.sin(roll), np.cos(roll)]])
    Rp = np.matrix([[np.cos(pitch), 0.0, np.sin(pitch)],[0.0, 1.0, 0.0],[-np.sin(pitch), 0.0, np.cos(pitch)]])
    Ry = np.matrix([[np.cos(yaw), -np.sin(yaw), 0.0],[np.sin(yaw), np.cos(yaw), 0.0],[0.0, 0.0, 1.0]])
    return Ry*Rp*Rr
    
def transformPointcloud(pcl, rotationMatrix, translationMatrix):
    return np.transpose(rotationMatrix*np.transpose(pcl))+translationMatrix