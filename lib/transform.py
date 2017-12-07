import numpy as np    

# -*- coding: utf-8 -*-
"""

Create rotation matrices, apply transformations to points

@author: Thorsten
"""    

"""
Rotation Matrices 3D
"""
def rotationMatrix_yaw(yaw):
    return np.matrix([[np.cos(yaw), -np.sin(yaw), 0.0],[np.sin(yaw), np.cos(yaw), 0.0],[0.0, 0.0, 1.0]])
    
def rotationMatrix_pitch(pitch):
    return np.matrix([[np.cos(pitch), 0.0, np.sin(pitch)],[0.0, 1.0, 0.0],[-np.sin(pitch), 0.0, np.cos(pitch)]])
    
def rotationMatrix_roll(roll):
    return np.matrix([[1.0, 0.0, 0.0],[0.0, np.cos(roll), -np.sin(roll)],[0.0, np.sin(roll), np.cos(roll)]])

def rotationMatrix_ypr(yaw, pitch, roll):
    """
    Create rotation matrix for 3 rotations according to DIN8855 to rotate points
    from a global coordinate system to vehicle coordinate system
    
    Thanks to balzer82 (github)
    http://www.cbcity.de/tutorial-rotationsmatrix-und-quaternion-einfach-erklaert-in-din70000-zyx-konvention
    
    """
    return rotationMatrix_yaw(yaw)*rotationMatrix_pitch(pitch)*rotationMatrix_roll(roll) 

"""
Rotation Matrix 2D
"""
def rotationMatrix_2D(angle):
    return np.matrix([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])

"""
Apply transformation
"""
def translatePointcloud(pcl,translationMatrix):
    return pcl + translationMatrix
    
def rotatePointcloud(pcl,rotationMatrix):
    return pcl*np.transpose(rotationMatrix)

def transformPointcloud(pcl, rotationMatrix, translationMatrix):
    return pcl*np.transpose(rotationMatrix) + translationMatrix