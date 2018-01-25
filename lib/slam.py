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
from lib.Particle import Particle

def slatch(path,nrOfScans,bbPseudoRadius,l_occupied,l_free,l_min,l_max,resolution,nrParticle,stddPos,stddYaw,UID):
    print('Start slatch')    
    
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
#try:
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
    # grid
    plt.imsave(path+'/grid_'+UID+'.png',grid[:,:],cmap='binary')
    # log file with parameter
    
    # trajectory 
    np.savetxt(path+'trajectory_'+UID+'.txt',trajectory,delimiter=',')
    # error
    np.savetxt(path+'error_'+UID+'.txt',np.hstack((distanceGT,errorPos)),delimiter=',')
    
    return errorPos.sum()

#except:
    print('There was an exception, return -1')
    return -1.0
        

def slatchRE(dataset,zCutMin,zCutMax,bbPseudoRadius,l_occupied,l_free,l_min,l_max,length,width,resolution,nrParticle,stddPos,stddYaw):
    """
    Create empty GRID [m] and initialize it with 0, this Log-Odd 
    value is maximum uncertainty (P = 0.5)
    """
    grid = np.zeros((int(length/resolution),
                 int(width/resolution)),
                 order='C')
    offset = np.array([[length/2.0,width/2.0]])
    startPos = np.array([[0,0]])

    """
    Process all Scans
    """
    particles = []
    traj = []
    trajGT = []
    evaTraj = []
    evaTrajGT = []
    evaDist = []
    evaDist.append(np.matrix([0.0]))
    evaDistGT = []
    evaDistGT.append(np.matrix([0.0]))
    yaw = 0
    T = np.matrix([0.0,0.0])
    dYaw = 0
    dT = np.matrix([0.0,0.0])
    nr = 0

    it = dataset.oxts    
    
    for scan in dataset.velo:
        particles.clear()
        t0 = time.time()
        print(nr)    
        pose = next(it)    
    
        """
        Save Ground Truth Trajectory
        """
        pos_cartesian = utm.from_latlon(pose.packet.lat,pose.packet.lon)    
        TGT = np.matrix([pos_cartesian[0], pos_cartesian[1]])
        trajGT.append(TGT)
        temp = np.matrix([pos_cartesian[0], pos_cartesian[1],pose.packet.yaw])
        evaTrajGT.append(temp)
        """
        Filter pointcloud
        """
        # get PCL and delete intensity
        points = np.asarray(scan)
        points = np.delete(points,3,1)
        # limit range 
        points = filterPCL.filterBB(points,bbPseudoRadius,0.0,0.0)
        # get pcl where z-value is in a certain range, in vehicle plane!
        points = filterPCL.filterZ(points,zCutMin,zCutMax) 
         # translate PCL from velodyne coord system to GPS/IMU coord System
        R_Imu2Velod = np.matrix(dataset.calib.T_velo_imu[0:3,0:3])
        T_Imu2Velod = np.matrix(dataset.calib.T_velo_imu[0:3,3])
        points = transform.transformPointcloud(points,np.linalg.inv(R_Imu2Velod),-T_Imu2Velod) 
        # project points to xy-plane (delete z values)
        points = np.delete(points,2,1)
        
        """
        First measurement: get initialisation
        """
        if nr == 0:
            startPos = np.array([[pos_cartesian[0], pos_cartesian[1]]])
            T=TGT
            yaw = pose.packet.yaw       
        """
        All other measurements: Get position with basic particle filter, match 
        measurement to GRID
        """
        bestEstimateParticle = Particle(T[0,0],T[0,1],yaw,0)
        if nr!=0:
            # Create particles
            for _ in range(0,500):
                x = np.random.normal(T[0,0]+dT[0,0],stddPos)
                y = np.random.normal(T[0,1]+dT[0,1],stddPos)
                tempYaw = np.random.normal(yaw+dYaw,stddYaw)
                p = Particle(x,y,tempYaw,1)
                particles.append(p)
            # Weight particles
            weightMin = math.inf
            for p in particles:
                pointsT = transform.rotatePointcloud(points,transform.rotationMatrix_2D(p.yaw))
                pointsT = transform.translatePointcloud(pointsT,np.matrix([p.x,p.y]))
                p.weight = mapping.scan2mapDistance(grid,pointsT,startPos-offset,resolution)
                if p.weight < weightMin:
                    weightMin = p.weight
            # Norm weights
            weightSum = 0
            for p in particles:
                p.weight -= weightMin
                weightSum += p.weight
            for p in particles:
                p.weight /= weightSum
            # Sort particles with weight
            particles.sort(key = lambda Particle: Particle.weight,reverse=True)
            # get only the 10 best particles
            bestParticles = particles[:10]
            
            print('Resample')
            particles.clear()
            nrParticleResample = 50
            for p in bestParticles:
                for _ in range(0,nrParticleResample):
                    x = np.random.normal(p.x,stddPos/6)
                    y = np.random.normal(p.y,stddPos/6)
                    yaw = np.random.normal(p.yaw,stddYaw/3)
                    pp = Particle(x,y,yaw,1)
                    particles.append(pp)    
            # Weight particles
            for p in particles:
                pointsT = transform.rotatePointcloud(points,transform.rotationMatrix_2D(p.yaw))
                pointsT = transform.translatePointcloud(pointsT,np.matrix([p.x,p.y]))
                p.weight = mapping.scan2mapDistance(grid,pointsT,startPos-offset,resolution)

            # Sort particles with weight
            particles.sort(key = lambda Particle: Particle.weight,reverse=True)
            
            
            # Choose best particle
            bestEstimateParticle = particles[0]
            dT = np.matrix([bestEstimateParticle.x,bestEstimateParticle.y])-T
            dYaw = bestEstimateParticle.yaw-yaw
            T = np.matrix([bestEstimateParticle.x,bestEstimateParticle.y])
            yaw = bestEstimateParticle.yaw
            # calculate distance traveled
            dist = evaDist[-1]+np.sqrt( np.multiply(dT[0,0], dT[0,0]) + np.multiply( dT[0,1], dT[0,1]))
            evaDist.append(dist)
            distGT = evaDist[-1]+np.sqrt( np.multiply(trajGT[-1][0,0]-trajGT[-2][0,0],trajGT[-1][0,0]-trajGT[-2][0,0])
                                         +np.multiply(trajGT[-1][0,1]-trajGT[-2][0,1],trajGT[-1][0,1]-trajGT[-2][0,1]))
            evaDistGT.append(distGT)
              
        """
        Add measurement to grid and save trajectory point
        """
        # rotate points with yaw from startPose   
        points = transform.rotatePointcloud(points, transform.rotationMatrix_2D(yaw))
        # translate points to global coord system with startPose      
        points = transform.translatePointcloud(points,T)
        # add measurement to grid
        mapping.addMeasurement(grid,points[:,0],points[:,1],T,startPos-offset,
                               resolution,l_occupied,l_free,l_min,l_max)
        # save traj point
        traj.append(T)
        temp = np.matrix([bestEstimateParticle.x,bestEstimateParticle.y,bestEstimateParticle.yaw])
        evaTraj.append(temp)
    
        #if nr == 153:
        #    break
        nr+=1
        dt = time.time() - t0
        print('Duration: %.2f' % dt)
    
    """
    Evaluate Trajectory
    """
    evaTraj = np.vstack(evaTraj)
    evaTrajGT = np.vstack(evaTrajGT)
    evaDist = np.vstack(evaDist)
    evaDistGT = np.vstack(evaDistGT)

    error = np.sqrt( np.multiply(evaTraj[:,0]-evaTrajGT[:,0],evaTraj[:,0]-evaTrajGT[:,0])+np.multiply(evaTraj[:,1]-evaTrajGT[:,1],evaTraj[:,1]-evaTrajGT[:,1]) )    
    
    evaluation = np.hstack((evaDistGT,error,evaTraj[:,2]-evaTrajGT[:,2]))
    return evaluation