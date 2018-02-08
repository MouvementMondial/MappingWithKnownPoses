# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:28:58 2018

@author: Thorsten
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import math

from lib import mapping
from lib import filterPCL
from lib import posEstimation

def slatch(path,nrOfScans,bbPseudoRadius,l_occupied,l_free,l_min,l_max,resolution,nrParticle,stddPos,stddYaw,UID):
    #print('Start slatch')    
    
    """
    Parameter
    """    
    zCutMin = -0.05
    zCutMax = 0.05    
    
    """
    Load data 
    (use script 'KittiExport' to create this file formate from KITTI raw data or
    KITTI odometry data)
    """
    startPose = np.loadtxt(path+'firstPose.txt',delimiter=',')
    groundTruth = np.asmatrix(np.loadtxt(path+'groundTruth.txt',delimiter=','))
    
    """
    Create empty GRID [m] and initialize it with 0, this Log-Odd 
    value is maximum uncertainty (P = 0.5)
    Use the ground truth trajectory to calculate width and length of the grid
    """
    length = groundTruth[:,0].max()-groundTruth[:,0].min()+200.0
    width = groundTruth[:,1].max()-groundTruth[:,1].min()+200.0
    
    grid = np.zeros((int(length/resolution),int(width/resolution)),order='C')
    offset = np.array([math.fabs(groundTruth[:,0].min()-100.0),math.fabs(groundTruth[:,1].min()-100.0),0.0]) # x y yaw
    
    #print('Length: '+str(length))
    #print('Width: '+str(width))    
    
    """
    Declare some varibles 
    """
    # save estimated pose (x y yaw) here
    trajectory = [] 
    # save the estimated pose here (mouvement model with deltaPose)
    estimatePose = np.matrix([startPose[0],startPose[1],startPose[2]])
    
    """
    Process all Scans
    """
#try:
    for ii in range(0,nrOfScans+1):
        t0 = time.time() # measure time
    
        """
        Load and filter pointcloud
        """
        pointcloud = np.load(path+'pointcloudNP_'+str(ii)+'.npy')    
        # limit range of velodyne laserscaner
        pointcloud = filterPCL.filterBB(pointcloud,bbPseudoRadius,0.0,0.0)
        # get points where z-value is in a certain range, in vehicle plane!
        pointcloud = filterPCL.filterZ(pointcloud,zCutMin,zCutMax)
        # project points to xy-plane (delete z-values)
        pointcloud = np.delete(pointcloud,2,1)
    
        """
        First measurement: map with initial position

        Second measurement: There is no information about the previous movement 
        of the robot, use the advanced particlefilter with a lot of particles
        to find the first estimate. 
        Use the usal particlefilter to find the position and add the measurement 
        with this position to the grid.
    
        All other measurements: Use the precious movement of the robot to calculate
        the estimated position.  Use the usal particlefilter to find the position 
        and add the measurement with this position to the grid.
        """
        if ii != 0: # if it is not the first measurement
            """
            Estimate the next pose
            """
            if ii == 1: # if it is the second measurement: Use particlefilter to find estimate
                estimatePose = posEstimation.fitScan2Map(grid,pointcloud,50000,0,0,
                                                         estimatePose,stddPos*10,stddYaw*10,
                                                         startPose,offset,resolution)              
                        
            else: # it is not the first or second measurement, use movement model
                deltaPose = trajectory[-1]-trajectory[-2]
                estimatePose = trajectory[-1]+deltaPose
        
            """
            Fit scan to map
            """
            estimatePose = posEstimation.fitScan2Map(grid,pointcloud,500,10,50,
                                                     estimatePose,stddPos,stddYaw,
                                                     startPose,offset,resolution)

        """
        Add measurement to grid
        """
        # Transform pointcloud to best estimate
        # rotation
        R = np.matrix([[np.cos(estimatePose[0,2]),-np.sin(estimatePose[0,2])],
                       [np.sin(estimatePose[0,2]),np.cos(estimatePose[0,2])]])
        pointcloud = pointcloud * np.transpose(R)
        # translation
        pointcloud = pointcloud + np.matrix([estimatePose[0,0],
                                             estimatePose[0,1]])
        # Add measurement to grid
        mapping.addMeasurement(grid,pointcloud[:,0],pointcloud[:,1],
                               np.matrix([estimatePose[0,0], estimatePose[0,1],0.0]),
                               startPose-offset,resolution,l_occupied,l_free,l_min,l_max)
        # Add the used pose to the trajectory
        trajectory.append(estimatePose)
        #print('Scan '+str(ii)+' processed: '+str(time.time()-t0)+'s')
        
    # calculate evaluation
    # position error
    trajectory = np.vstack(trajectory)
    errorPos = np.sqrt( np.multiply(trajectory[:,0]-groundTruth[:,0],trajectory[:,0]-groundTruth[:,0])
               +np.multiply(trajectory[:,1]-groundTruth[:,1],trajectory[:,1]-groundTruth[:,1]))
    # traveled distance ground truth
    temp = np.vstack((groundTruth[0,:],groundTruth))
    temp = np.delete(temp,(temp.shape[0]-1), axis=0)
    distanceGT = np.sqrt( np.multiply(temp[:,0]-groundTruth[:,0],temp[:,0]-groundTruth[:,0])
                         +np.multiply(temp[:,1]-groundTruth[:,1],temp[:,1]-groundTruth[:,1]))
    distanceGT = np.transpose(np.cumsum(distanceGT))    
    
    """
    Save results
    """
    UID = str(time.time())+'test'
    # grid
    plt.imsave(path+'/grid_'+UID+'.png',grid[:,:],cmap='binary')
    # log file with parameter
    logfile = open(path+'/log_'+UID+'.txt','w')
    logfile.write('Slatch with resampling'+'\n')
    logfile.write('Measurement: '+path+'\n')
    logfile.write('UID: '+UID+'\n')
    logfile.write('bbPseudoRadius = '+str(bbPseudoRadius)+'\n')
    logfile.write('l_occupied = '+str(l_occupied)+'\n')
    logfile.write('l_free = '+str(l_free)+'\n')
    logfile.write('l_min = '+str(l_min)+'\n')
    logfile.write('l_max = '+str(l_max)+'\n')
    logfile.write('resolution = '+str(resolution)+'\n')
    logfile.write('stddPos = '+str(stddPos)+'\n')
    logfile.write('stddYaw = '+str(stddYaw)+'\n')
    logfile.write('errorSum = '+str(errorPos.sum())+'\n')
    logfile.close()
    # trajectory 
    np.savetxt(path+'trajectory_'+UID+'.txt',trajectory,delimiter=',')
    # error
    np.savetxt(path+'error_'+UID+'.txt',np.hstack((distanceGT,errorPos)),delimiter=',')
    
    return trajectory

#except:
    print('There was an exception, return -1')
    return -1.0
        

def slatchNoResample(path,nrOfScans,bbPseudoRadius,l_occupied,l_free,l_min,l_max,resolution,nrParticle,stddPos,stddYaw,UID):
    #print('Start slatch')    
    
    """
    Parameter
    """    
    zCutMin = -0.05
    zCutMax = 0.05    
    
    """
    Load data 
    (use script 'KittiExport' to create this file formate from KITTI raw data or
    KITTI odometry data)
    """
    startPose = np.loadtxt(path+'firstPose.txt',delimiter=',')
    groundTruth = np.asmatrix(np.loadtxt(path+'groundTruth.txt',delimiter=','))
    
    """
    Create empty GRID [m] and initialize it with 0, this Log-Odd 
    value is maximum uncertainty (P = 0.5)
    Use the ground truth trajectory to calculate width and length of the grid
    """
    length = groundTruth[:,0].max()-groundTruth[:,0].min()+200.0
    width = groundTruth[:,1].max()-groundTruth[:,1].min()+200.0
    
    grid = np.zeros((int(length/resolution),int(width/resolution)),order='C')
    offset = np.array([math.fabs(groundTruth[:,0].min()-100.0),math.fabs(groundTruth[:,1].min()-100.0),0.0]) # x y yaw
    
    #print('Length: '+str(length))
    #print('Width: '+str(width))    
    
    """
    Declare some varibles 
    """
    # save estimated pose (x y yaw) here
    trajectory = [] 
    # save the estimated pose here (mouvement model with deltaPose)
    estimatePose = np.matrix([startPose[0],startPose[1],startPose[2]])
    
    """
    Process all Scans
    """
#try:
    for ii in range(0,nrOfScans+1):
        t0 = time.time() # measure time
    
        """
        Load and filter pointcloud
        """
        pointcloud = np.load(path+'pointcloudNP_'+str(ii)+'.npy')    
        # limit range of velodyne laserscaner
        pointcloud = filterPCL.filterBB(pointcloud,bbPseudoRadius,0.0,0.0)
        # get points where z-value is in a certain range, in vehicle plane!
        pointcloud = filterPCL.filterZ(pointcloud,zCutMin,zCutMax)
        # project points to xy-plane (delete z-values)
        pointcloud = np.delete(pointcloud,2,1)
    
        """
        First measurement: map with initial position

        Second measurement: There is no information about the previous movement 
        of the robot, use the advanced particlefilter with a lot of particles
        to find the first estimate. 
        Use the usal particlefilter to find the position and add the measurement 
        with this position to the grid.
    
        All other measurements: Use the precious movement of the robot to calculate
        the estimated position.  Use the usal particlefilter to find the position 
        and add the measurement with this position to the grid.
        """
        if ii != 0: # if it is not the first measurement
            """
            Estimate the next pose
            """
            if ii == 1: # if it is the second measurement: Use particlefilter to find estimate
                estimatePose = posEstimation.fitScan2Map(grid,pointcloud,50000,0,0,
                                                         estimatePose,stddPos*10,stddYaw*10,
                                                         startPose,offset,resolution)              
                        
            else: # it is not the first or second measurement, use movement model
                deltaPose = trajectory[-1]-trajectory[-2]
                estimatePose = trajectory[-1]+deltaPose
        
            """
            Fit scan to map
            """
            estimatePose = posEstimation.fitScan2Map(grid,pointcloud,1000,0,0,
                                                     estimatePose,stddPos,stddYaw,
                                                     startPose,offset,resolution)

        """
        Add measurement to grid
        """
        # Transform pointcloud to best estimate
        # rotation
        R = np.matrix([[np.cos(estimatePose[0,2]),-np.sin(estimatePose[0,2])],
                       [np.sin(estimatePose[0,2]),np.cos(estimatePose[0,2])]])
        pointcloud = pointcloud * np.transpose(R)
        # translation
        pointcloud = pointcloud + np.matrix([estimatePose[0,0],
                                             estimatePose[0,1]])
        # Add measurement to grid
        mapping.addMeasurement(grid,pointcloud[:,0],pointcloud[:,1],
                               np.matrix([estimatePose[0,0], estimatePose[0,1],0.0]),
                               startPose-offset,resolution,l_occupied,l_free,l_min,l_max)
        # Add the used pose to the trajectory
        trajectory.append(estimatePose)
        #print('Scan '+str(ii)+' processed: '+str(time.time()-t0)+'s')
        
    # calculate evaluation
    # position error
    trajectory = np.vstack(trajectory)
    errorPos = np.sqrt( np.multiply(trajectory[:,0]-groundTruth[:,0],trajectory[:,0]-groundTruth[:,0])
               +np.multiply(trajectory[:,1]-groundTruth[:,1],trajectory[:,1]-groundTruth[:,1]))
    # traveled distance ground truth
    temp = np.vstack((groundTruth[0,:],groundTruth))
    temp = np.delete(temp,(temp.shape[0]-1), axis=0)
    distanceGT = np.sqrt( np.multiply(temp[:,0]-groundTruth[:,0],temp[:,0]-groundTruth[:,0])
                         +np.multiply(temp[:,1]-groundTruth[:,1],temp[:,1]-groundTruth[:,1]))
    distanceGT = np.transpose(np.cumsum(distanceGT))    
    
    """
    Save results
    """
    UID = str(time.time())
    # grid
    plt.imsave(path+'/grid_'+UID+'.png',grid[:,:],cmap='binary')
    # log file with parameter
    logfile = open(path+'/log_'+UID+'.txt','w')
    logfile.write('Slatch with resampling'+'\n')
    logfile.write('Measurement: '+path+'\n')
    logfile.write('UID: '+UID+'\n')
    logfile.write('bbPseudoRadius = '+str(bbPseudoRadius)+'\n')
    logfile.write('l_occupied = '+str(l_occupied)+'\n')
    logfile.write('l_free = '+str(l_free)+'\n')
    logfile.write('l_min = '+str(l_min)+'\n')
    logfile.write('l_max = '+str(l_max)+'\n')
    logfile.write('resolution = '+str(resolution)+'\n')
    logfile.write('stddPos = '+str(stddPos)+'\n')
    logfile.write('stddYaw = '+str(stddYaw)+'\n')
    logfile.write('errorSum = '+str(errorPos.sum())+'\n')
    logfile.close()
    # trajectory 
    np.savetxt(path+'trajectory_'+UID+'.txt',trajectory,delimiter=',')
    # error
    np.savetxt(path+'error_'+UID+'.txt',np.hstack((distanceGT,errorPos)),delimiter=',')
    
    return trajectory

#except:
    print('There was an exception, return -1')
    return -1.0