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

'''
Load data
'''

# raw
basedir = 'C:/KITTI'
date = '2011_09_26'
drive = '0117'
dataset = pykitti.raw(basedir, date, drive, imformat='cv2')
it = dataset.oxts
"""
# odometry
basedir = 'D:/KITTI/odometry/dataset'
sequence = '01'
dataset = pykitti.odometry(basedir, sequence)
"""
print('Hallo')

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
length = 500.0
width = 500.0
resolution = 0.25

grid = np.zeros((int(length/resolution),
                 int(width/resolution)),
                 order='C')
offset = np.array([[length/2.0,width/2.0]])
startPos = np.array([[0,0]])

"""
Parameter particle Filter
"""
nrParticle = 1000
stddPos = 0.1
stddYaw = 0.025
particles = []

"""
Process all Scans
"""
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
dT = np.matrix([0.0,0.7])
nr = 0

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
    # translate PCL from velodyne coord system to GPS/IMU coord System
    R_Imu2Velod = np.matrix(dataset.calib.T_velo_imu[0:3,0:3])
    T_Imu2Velod = np.matrix(dataset.calib.T_velo_imu[0:3,3])
    points = transform.transformPointcloud(points,np.linalg.inv(R_Imu2Velod),-T_Imu2Velod) 
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
    bestEstimateParticle = Particle(T[0,0],T[0,1],yaw,0)
    if nr!=0:
        # Create particles
        for _ in range(0,nrParticle):
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
        '''
        # Norm weights
        weightSum = 0
        for p in particles:
            p.weight -= weightMin
            weightSum += p.weight
        for p in particles:
            p.weight /= weightSum
        '''
        # Sort particles with weight
        particles.sort(key = lambda Particle: Particle.weight,reverse=True)
        # get only the 10 best particles
        bestParticles = particles[:5]

        '''
        print('Resample')
        xMean = sum(_.x for _ in bestparticles)/len(bestparticles)
        yMean = sum(_.y for _ in bestparticles)/len(bestparticles)
        yawMean = sum(_.yaw for _ in bestparticles)/len(bestparticles)        
        bestEstimateParticle = Particle(xMean,yMean,yawMean,1.0)
        '''
        
        """
        print('Resample')
        particles.clear()
        nrParticleResample = 100
        for p in bestParticles:
            for _ in range(0,nrParticleResample):
                x = np.random.normal(p.x,stddPos/1)
                y = np.random.normal(p.y,stddPos/1)
                yaw = np.random.normal(p.yaw,stddYaw/1)
                pp = Particle(x,y,yaw,1)
                particles.append(pp)    
        # Weight particles
        for p in particles:
            pointsT = transform.rotatePointcloud(points,transform.rotationMatrix_2D(p.yaw))
            pointsT = transform.translatePointcloud(pointsT,np.matrix([p.x,p.y]))
            p.weight = mapping.scan2mapDistance(grid,pointsT,startPos-offset,resolution)

        # Sort particles with weight
        particles.sort(key = lambda Particle: Particle.weight,reverse=True)
        """
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
    
    #if nr == 1:
    #     break
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
plt.title('Slatch')

# save map        
io.writeGrid2Img(grid,date+'_'+drive+'_grid_'+str(nr)+'slatch.png')
traj = np.vstack(traj)
np.savetxt('trajP.txt',traj,delimiter=',',fmt='%1.3f')
trajGT = np.vstack(trajGT)
np.savetxt('trajGT.txt',traj,delimiter=',',fmt='%1.3f')

"""
Evaluate Trajectory
"""
evaTraj = np.vstack(evaTraj)
evaTrajGT = np.vstack(evaTrajGT)
evaDist = np.vstack(evaDist)
evaDistGT = np.vstack(evaDistGT)

error = np.sqrt( np.multiply(evaTraj[:,0]-evaTrajGT[:,0],evaTraj[:,0]-evaTrajGT[:,0])+np.multiply(evaTraj[:,1]-evaTrajGT[:,1],evaTraj[:,1]-evaTrajGT[:,1]) )
            
plt.figure(1)
plt.subplot(211)
plt.plot(evaDistGT,error)
plt.title('Abweichung Position zu Ground Truth')
plt.xlabel('Distanz GT [m]')
plt.ylabel('Abweichung [m]')

plt.subplot(212)
errYaw = np.asarray((evaTraj[:,2]-evaTrajGT[:,2])/math.pi*180)
errYaw[errYaw>=180.0] -= 360.0
errYaw[errYaw<=-180.0] += 360.0
plt.plot(evaDistGT,errYaw)
plt.title('Abweichung Yaw zu Ground Truth')
plt.xlabel('Distanz GT [m]')
plt.ylabel('Abweichung [°]')

evaluation = np.hstack((evaDistGT,error,evaTraj[:,2]-evaTrajGT[:,2]))

np.savetxt('eva_'+date+'_'+drive+'_'+'1000.txt',evaluation,delimiter=',',fmt='%1.5f')