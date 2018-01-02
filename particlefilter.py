# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:14:52 2017

0.1s bei 50kmh

v = s/t
s= v/t
s = 50/3.6/0.1
s = 50/36 = 1.388

20° = 20/180*3.1415 = 0.35


@author: Thorsten
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import bisect

from lib.Particle import Particle
from lib import mapping
from lib import io
from lib import transform

import random

print('Start')

scanNr = 58 # start bei 1

folder = 'C:/KITTI/Testdaten/partikel_20110926_0005/'

grid = io.readTxt2Grid(folder+'grid_'+str(scanNr)+'.txt')
pcl = io.readPointcloud2xyz(folder+'pcl_'+str(scanNr)+'.txt')
pcl = np.delete(pcl,2,1)
traj = io.readTraj2xyz(folder+'traj.txt')



offset = np.array([[-50,-50]])
offsetTraj = np.array([[traj[0,0],traj[0,1]]])
resolution = 0.1

particles = []

nrParticle = 1000
xyd = np.zeros((3,nrParticle))

print('Create particles')
for _ in range(0,nrParticle):
    x = np.random.normal(traj[scanNr,0]-offsetTraj[0,0],1)
    y = np.random.normal(traj[scanNr,1]-offsetTraj[0,1],1)
    #yaw = -0.5436626732051#0027
    yaw = traj[scanNr-1,2]
    yawERR = 0.00
    yaw = random.uniform(-yawERR+yaw,yawERR+yaw)
    p = Particle(x,y,yaw,1)
    particles.append(p)

print('Weight particles')
weightMin = math.inf
for p in particles:
    pclt = transform.rotatePointcloud(pcl,transform.rotationMatrix_2D(p.yaw))
    pclt = transform.translatePointcloud(pclt,np.matrix([p.x,p.y]))
    p.weight = mapping.scan2mapDistance(grid,pclt,offset,resolution)   
    if p.weight < weightMin:
        weightMin = p.weight

print('Norm weights')
weightSum = 0
for p in particles:
    p.weight -= weightMin    
    weightSum += p.weight
for p in particles:
    p.weight /= weightSum
    
print('Resample')
particles.sort(key = lambda Particle: Particle.weight)
scDsum = []
temp = 0
for p in particles:
    temp += p.weight
    scDsum.append(temp)
newParticles = []
for _ in range(0,nrParticle):
    newParticles.append(particles[bisect.bisect_left(scDsum, random.uniform(0,1))])
    
print('show')
printBest = 15

particles.sort(key = lambda Particle: Particle.weight,reverse=True)
newParticles.sort(key = lambda Particle: Particle.weight,reverse=True)
xyyd = [[p.x,p.y,p.yaw,p.weight] for p in particles]
scX,scY,scYaw,scD = zip(*xyyd)

# plot grid
plt.figure(0)
plt.imshow(grid[:,:], interpolation ='none', cmap = 'binary')

# plot particle
plt.scatter((scY-offset[0,1])/resolution,(scX-offset[0,0])/resolution,c='r',s=10,edgecolors='none')
plt.scatter((scY[0:printBest]-offset[0,1])/resolution,(scX[0:printBest]-offset[0,0])/resolution,c='y',s=40)
plt.scatter((scY[0]-offset[0,1])/resolution,(scX[0]-offset[0,0])/resolution,c='c',s=80, marker='*')

# plot pcl
pclt = transform.rotatePointcloud(pcl,transform.rotationMatrix_2D(scYaw[0]))
pclt = transform.translatePointcloud(pclt,np.matrix([scX[0],scY[0]]))
plt.scatter(([pclt[:,1]]-offset[0,0])/resolution,([pclt[:,0]]-offset[0,1])/resolution,c='m')

# plot trajectory
plt.scatter(([traj[:,1]]-offsetTraj[0,1]-offset[0,1])/resolution,
            ([traj[:,0]]-offsetTraj[0,0]-offset[0,0])/resolution,
            c='g',s=40,edgecolors='none')
plt.scatter(([traj[scanNr,1]]-offsetTraj[0,1]-offset[0,1])/resolution,
            ([traj[scanNr,0]]-offsetTraj[0,0]-offset[0,0])/resolution,
            c='g',s=80)

#plt.figure(1)
#plt.hist(scYaw[0:printBest])
#plt.figure(2)
#plt.hist(scD[0:printBest])
#plt.figure(3)
#plt.plot(np.arange(0,nrParticle,1),scDsum,marker='o')
#xyyd = [[p.x,p.y,p.yaw,p.weight] for p in newParticles]
#scX,scY,scYaw,scD = zip(*xyyd)
#plt.figure(4)
#plt.imshow(grid[:,:], interpolation ='none', cmap = 'binary')
#plt.scatter((scX-offset[0,0])/resolution,(scY-offset[0,1])/resolution,c='r')
#plt.scatter((scX[0:printBest]-offset[0,0])/resolution,(scY[0:printBest]-offset[0,1])/resolution,c='y',s=40)
print('finish')
