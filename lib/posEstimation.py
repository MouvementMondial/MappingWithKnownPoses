# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:57:46 2018

@author: Thorsten
"""

import numpy as np
from numba import jit
from lib.Particle import Particle

@jit(nopython=True)
def scan2mapDistance(grid,pcl,offset,resolution):
    distance = 0;
    for i in range(pcl.shape[0]):
        # round points to cells
        xi = int ( (pcl[i,0]-offset[0]) / resolution )
        yi = int ( (pcl[i,1]-offset[1]) / resolution )
        #if grid[xi,yi] > 0:    
        distance += grid[xi,yi]
    return distance
    
def fitScan2Map(grid,pcl,nrParticle1,nrParticleResample,nrParticle2,
                firstEstimate,stddPos,stddYaw,
                startPose,offset,resolution):
    
    # save the particles of the particle filter here
    particles = []
    
    """
    Particle filter with nrParticle1 Particles for first estimate
    """                
    # Create particles
    for _ in range(0,nrParticle1):
        particles.append(Particle(np.random.normal(firstEstimate[0],stddPos),
                                  np.random.normal(firstEstimate[1],stddPos),
                                  np.random.normal(firstEstimate[2],stddYaw),
                                  0.0))
    # Weight particles
    for p in particles:
        # transform pointcloud to estimated pose
        # rotation
        R = np.matrix([[np.cos(p.yaw),-np.sin(p.yaw)],
                       [np.sin(p.yaw),np.cos(p.yaw)]])
        pclTransformed = pcl * np.transpose(R)
        # translation            
        pclTransformed = pclTransformed + np.matrix([p.x,p.y])
        # weight this solution
        p.weight = scan2mapDistance(grid,pointcloudTransformed,
                                    startPose-offset,resolution)
    # sort particles by weight (best first)
    particles.sort(key = lambda Particle: Particle.weight,reverse=True)
    
    """
    Return best particle if there is no resampling
    """
    if nrParticleResample == 0:
        return particles[0]
   
    """
    Look in the neighborhood of the best particles for better solutions
    """
    # get only the best particles
    bestParticles = particles[:nrParticleResample]
    # delete the old particles
    particles.clear()
    
    # for each of the good particles create new particles
    for p in bestParticles:
        for _ in range(0,nrParticle2):
                particles.append(Particle(np.random.normal(p.x,stddPos/6.0),
                                          np.random.normal(p.y,stddPos/6.0),
                                          np.random.normal(p.yaw,stddYaw/3.0),
                                          0.0))
    # Weight particles
    for p in particles:
        # transform pointcloud to estimated pose
        # rotation
        R = np.matrix([[np.cos(p.yaw),-np.sin(p.yaw)],
                       [np.sin(p.yaw),np.cos(p.yaw)]])
        pclTransformed = pcl * np.transpose(R)
        # translation            
        pclTransformed = pclTransformed + np.matrix([p.x,p.y])
        # weight this solution
        p.weight = scan2mapDistance(grid,pointcloudTransformed,
                                    startPose-offset,resolution)
    # sort particles by weight (best first)
    particles.sort(key = lambda Particle: Particle.weight,reverse=True)
    
    """
    Return the best particle
    """
    return particles[0]
        