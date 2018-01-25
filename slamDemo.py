# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:14:35 2018

@author: Thorsten
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import math

from lib import mapping
from lib import filterPCL
from lib.Particle import Particle

'''
Load data 
(use script 'KittiExport' to create this file formate from KITTI raw data or
KITTI odometry data)
'''
path = 'D:/KITTI/odometry/dataset/04_export/'
nrOfScans = 270
startPose = np.loadtxt(path+'firstPose.txt',delimiter=',')
groundTruth = np.asmatrix(np.loadtxt(path+'groundTruth.txt',delimiter=','))

"""
Parameter pointcloud Filter
"""
# The IMU is mounted at 0.63m height
zCutMin = -0.05
zCutMax = 0.05
bbPseudoRadius = 50

"""
Parameter mapping from Octomap paper
"""
l_occupied = 0.85
l_free = -0.4
l_min = -2.0
l_max = 3.5

"""
Parameter particle Filter
"""
stddPos = 0.1
stddYaw = 0.20

"""
Create empty GRID [m] and initialize it with 0, this Log-Odd 
value is maximum uncertainty (P = 0.5)
Use the ground truth trajectory to calculate width and length of the grid
"""
length = groundTruth[:,0].max()-groundTruth[:,0].min()+200.0
width = groundTruth[:,1].max()-groundTruth[:,1].min()+200.0
resolution = 0.10

grid = np.zeros((int(length/resolution),int(width/resolution)),order='C')
offset = np.array([math.fabs(groundTruth[:,0].min()-100.0),math.fabs(groundTruth[:,1].min()-100.0),0.0]) # x y yaw

print('Length: '+str(length))
print('Width: '+str(width))

"""
Declare some varibles 
"""
# save the particles of the particle filter here
particles = [] 
# save estimated pose (x y yaw) here
trajectory = [] 
# save the best pose for each iteration here, initialisation with first pose
bestEstimateParticle  = Particle(startPose[0],startPose[1],startPose[2],1.0)
# save the difference of two poses here
deltaPose = np.matrix([0,0,0])
# save the estimated pose here (mouvement model with deltaPose)
estimatePose = np.matrix([0,0,0])

"""
Process all Scans
"""
try:
    for ii in range(0,nrOfScans+1):
        t0 = time.time() # measure time
        particles.clear() # new particles every scan
    
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
            Estimate the next pose. The robot will move as it did in the last step.
            """
            if ii == 1: # if it is the second measurement: Use particlefilter to find estimate
                for _ in range(0,50000):
                    particles.append(Particle(np.random.normal(startPose[0],stddPos*10),
                                              np.random.normal(startPose[1],stddPos*10),
                                              np.random.normal(startPose[2],stddYaw*10),
                                              0.0))
                # Weight particles
                for p in particles:
                    # transform pointcloud to estimated pose
                    # rotation
                    R = np.matrix([[np.cos(p.yaw),-np.sin(p.yaw)],
                                   [np.sin(p.yaw),np.cos(p.yaw)]])
                    pointcloudTransformed = pointcloud * np.transpose(R)
                    # translation            
                    pointcloudTransformed = pointcloudTransformed + np.matrix([p.x,p.y])
                    # weight this solution
                    p.weight = mapping.scan2mapDistance(grid,pointcloudTransformed,
                                                        startPose-offset,resolution)
                # sort particles by weight (best first)
                particles.sort(key = lambda Particle: Particle.weight,reverse=True)
                estimatePose = np.matrix([particles[0].x,particles[0].y,particles[0].yaw])      
                particles.clear()
        
            else: # it is not the first or second measurement, use movement model
                deltaPose = trajectory[-1]-trajectory[-2]
                estimatePose = trajectory[-1]+deltaPose
        
            """
            Particle filter with 500 Particles for first estimate
            """
            # Create first 500 Particles
            for _ in range(0,500):
                particles.append(Particle(np.random.normal(estimatePose[0,0],stddPos),
                                          np.random.normal(estimatePose[0,1],stddPos),
                                          np.random.normal(estimatePose[0,2],stddYaw),
                                          0.0))
                                      
            # Weight particles
            for p in particles:
                # transform pointcloud to estimated pose
                # rotation
                R = np.matrix([[np.cos(p.yaw),-np.sin(p.yaw)],
                               [np.sin(p.yaw),np.cos(p.yaw)]])
                pointcloudTransformed = pointcloud * np.transpose(R)
                # translation            
                pointcloudTransformed = pointcloudTransformed + np.matrix([p.x,p.y])
                # weight this solution
                p.weight = mapping.scan2mapDistance(grid,pointcloudTransformed,
                                                    startPose-offset,resolution)
            # sort particles by weight (best first)
            particles.sort(key = lambda Particle: Particle.weight,reverse=True)
        
            """
            Look in the neighborhood of the 10 best particles for better solutions
            """
            # get only the 10 best particles
            bestParticles = particles[:10]
            # delete the old particles
            particles.clear()
            # for each of the good particles, create 50 new particles
            for p in bestParticles:
                for _ in range(0,50):
                    particles.append(Particle(np.random.normal(p.x,stddPos/6.0),
                                              np.random.normal(p.y,stddPos/6.0),
                                              np.random.normal(p.yaw,stddYaw/3.0),
                                              0.0))
            #Weight particles
            for p in particles:
                # Transform pointcloud to estimated pose
                # rotation
                R = np.matrix([[np.cos(p.yaw),-np.sin(p.yaw)],
                               [np.sin(p.yaw),np.cos(p.yaw)]])
                pointcloudTransformed = pointcloud * np.transpose(R)
                # translation            
                pointcloudTransformed = pointcloudTransformed + np.matrix([p.x,p.y])
                # weight this solution
                p.weight = mapping.scan2mapDistance(grid,pointcloudTransformed,
                                                    startPose-offset,resolution)
            
            # sort particles by weight (best first)
            particles.sort(key = lambda Particle: Particle.weight,reverse=True)
        
            """
            Choose the best particle
            """
            bestEstimateParticle = particles[0]
    
        """
        Add measurement to grid
        """
        # Transform pointcloud to best estimate
        # rotation
        R = np.matrix([[np.cos(bestEstimateParticle.yaw),-np.sin(bestEstimateParticle.yaw)],
                       [np.sin(bestEstimateParticle.yaw),np.cos(bestEstimateParticle.yaw)]])
        pointcloud = pointcloud * np.transpose(R)
        # translation
        pointcloud = pointcloud + np.matrix([bestEstimateParticle.x,
                                             bestEstimateParticle.y])
        # Add measurement to grid
        mapping.addMeasurement(grid,pointcloud[:,0],pointcloud[:,1],
                               np.matrix([bestEstimateParticle.x, bestEstimateParticle.y,0.0]),
                               startPose-offset,resolution,l_occupied,l_free,l_min,l_max)
        # Add the used pose to the trajectory
        trajectory.append(np.matrix([bestEstimateParticle.x,bestEstimateParticle.y,bestEstimateParticle.yaw]))
        print('Scan '+str(ii)+' processed: '+str(time.time()-t0)+'s')

except:
    print('There was an exception, show results:')

"""
Save results
"""
plt.imsave(path+'/grid.png',grid[:,:],cmap='binary')

"""
Plot
"""
# show grid
plt.figure(0)
plt.imshow(grid[:,:], interpolation ='none', cmap = 'binary')
# plot trajectory
trajectory = np.vstack(trajectory)
plt.scatter(([trajectory[:,1]]-startPose[1]+offset[1])/resolution,
            ([trajectory[:,0]]-startPose[0]+offset[0])/resolution,
            c='b',s=30,edgecolors='none', label = 'Trajektorie SLAM')

# plot ground truth trajectory
plt.scatter(([groundTruth[:,1]]-startPose[1]+offset[1])/resolution,
            ([groundTruth[:,0]]-startPose[0]+offset[0])/resolution,
            c='g',s=30,edgecolors='none', label = 'Trajektorie Ground Truth')
# plot legend
plt.legend(loc='upper left')

# calculate evaluation
# position error
errorPos = np.sqrt( np.multiply(trajectory[:,0]-groundTruth[:,0],trajectory[:,0]-groundTruth[:,0])
                   +np.multiply(trajectory[:,1]-groundTruth[:,1],trajectory[:,1]-groundTruth[:,1]))
# traveled distance ground truth
temp = np.vstack((groundTruth[0,:],groundTruth))
temp = np.delete(temp,(temp.shape[0]-1), axis=0)
distanceGT = np.sqrt( np.multiply(temp[:,0]-groundTruth[:,0],temp[:,0]-groundTruth[:,0])
                     +np.multiply(temp[:,1]-groundTruth[:,1],temp[:,1]-groundTruth[:,1]))
distanceGT = np.transpose(np.cumsum(distanceGT))

# show evaluation
plt.figure(1)
plt.plot(distanceGT,errorPos)
plt.title('Abweichung Position zu Ground Truth')
plt.xlabel('Distanz GT [m]')
plt.ylabel('Abweichung [m]')