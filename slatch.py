# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 08:45:40 2018

@author: Thorsten
"""

import numpy as np
import matplotlib.pyplot as plt
import pykitti
import utm
import time
import math

from lib import mapping
from lib import transform
from lib import io
from lib import filterPCL
from lib.Particle import Particle

basedir = 'C:/KITTI'
date = '2011_09_30'
drive = '0027'

dataset = pykitti.raw(basedir, date, drive, imformat='cv2')
it = dataset.oxts

"""
Parameter PCL
"""
# The IMU is mounted at 0.63m height
zCutMin = -0.05
zCutMax = 0.05

bbPseudoRadius = 30.0

"""
Parameter mapping from Octomap paper
"""
l_occupied = 0.85
l_free = -0.4
l_min = -2.0
l_max = 3.5


"""
Create empty GRID [m] and initialize it with 0, this Log-Odd 
value is maximum uncertainty (P = 0.5)
"""
length = 2000.0
width = 2000.0
resolution = 0.1

grid = np.zeros((int(length/resolution),
                 int(width/resolution)),
                 order='C')
offset = np.array([[length/2.0,width/2.0]])
startPos = np.array([[0,0]])

"""
Parameter particle Filter
"""
nrParticle = 10000
stddPos = 0.5
stddYaw = 0.1
particles = []

"""
Process all Scans
"""
traj = []
trajGT =[]
yaw = 0
T = np.matrix([0.0,0.0])
nr = 0
for scan in dataset.velo:
    t0 = time.time()
    print(nr)    
    pose = next(it)    
    
    """
    Save Ground Truth Trajectory
    """
    pos_cartesian = utm.from_latlon(pose.packet.lat,pose.packet.lon)    
    TGT = np.matrix([pos_cartesian[0], pos_cartesian[1]])
    trajGT.append(TGT)
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
    if nr!=0:
        # Create particles
        for _ in range(0,nrParticle):
            x = np.random.normal(T[0,0],stddPos)
            y = np.random.normal(T[0,1],stddPos)
            yaw = np.random.normal(pose.packet.yaw,stddYaw)
            p = Particle(x,y,yaw,1)
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
        # Choose best particle
        bestEstimateParticle = particles[0]
        T = np.matrix([bestEstimateParticle.x,bestEstimateParticle.y])
        yaw = bestEstimateParticle.yaw
              
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
    
    #if nr == 153:
    #    break
    particles.clear()
    nr+=1
    dt = time.time() - t0
    print('Duration: %.2f' % dt)

"""
Plot
"""
printBest = 10
# plot grid
plt.figure(0)
plt.imshow(grid[:,:], interpolation ='none', cmap = 'binary')
# plot particle
xyyd = [[p.x,p.y,p.yaw,p.weight] for p in particles]
scX,scY,scYaw,scD = zip(*xyyd)
plt.scatter((scY+offset[0,1]-startPos[0,1])/resolution,(scX+offset[0,0]-startPos[0,0])/resolution,c='r',s=10,edgecolors='none')
plt.scatter((scY[0:printBest]+offset[0,1]-startPos[0,1])/resolution,(scX[0:printBest]+offset[0,0]-startPos[0,0])/resolution,c='y',s=40)
plt.scatter((scY[0]+offset[0,1]-startPos[0,1])/resolution,(scX[0]+offset[0,0]-startPos[0,0])/resolution,c='c',s=80, marker='*')

# plot pcl
plt.scatter(([points[:,1]]+offset[0,1]-startPos[0,1])/resolution,([points[:,0]]+offset[0,0]-startPos[0,0])/resolution,c='m',edgecolors='none')

# plot trajectory
traj = np.vstack(traj)
plt.scatter(([traj[:,1]]-startPos[0,1]+offset[0,1])/resolution,
            ([traj[:,0]]-startPos[0,0]+offset[0,0])/resolution,
            c='g',s=80,edgecolors='none')
trajGT = np.vstack(trajGT)
plt.scatter(([trajGT[:,1]]-startPos[0,1]+offset[0,1])/resolution,
            ([trajGT[:,0]]-startPos[0,0]+offset[0,0])/resolution,
            c='w',s=40)


# save map        
io.writeGrid2Img(grid,'grid_'+str(nr)+'.png')
traj = np.vstack(traj)
np.savetxt('trajP.txt',traj,delimiter=',',fmt='%1.3f')
trajGT = np.vstack(trajGT)
np.savetxt('trajGT.txt',traj,delimiter=',',fmt='%1.3f')
