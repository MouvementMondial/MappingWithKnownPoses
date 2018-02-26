import numpy as np
import matplotlib.pyplot as plt

import time
import math

from lib import posEstimation
from lib import mapping
from lib import filterPCL


"""
Create pointcloud
"""
nrPoints = 200000
radiusMean = 30

angles = np.linspace(0,2*np.pi,nrPoints) + np.random.rand(nrPoints)
radius = np.random.rand(nrPoints)+radiusMean-0.5

x = np.cos(angles)*radius
y = np.sin(angles)*radius
z = np.random.rand(nrPoints)
pcl = np.transpose(np.array([x,y,z]))

np.save('samplePCL',pcl)

#plt.figure(1)
#plt.scatter(x,y)

"""
Parameter pointcloud Filter
"""
# The LIDAR is mounted at 1.73m height
zCutMin = 0.5
zCutMax = 0.525
bbPseudoRadius = 1000

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
stddPos = 0.2
stddYaw = 0.025

"""
Create grid
"""
length = 1000
resolution = 0.25

grid = np.zeros((int(length/resolution),int(length/resolution)),order='C')
estimatePose = np.matrix([500.0,500.0,0.0])

offset = np.array([0.0,
                   0.0,
                   0.0])
                   
startPose = offset

nrOfRuns = 100

"""
Add measurement to grid
"""
pointcloud = np.load('samplePCL.npy')
pointcloud = filterPCL.filterBB(pointcloud,bbPseudoRadius,0.0,0.0)
pointcloud = filterPCL.filterZ(pointcloud,zCutMin,zCutMax)
pointcloud = np.delete(pointcloud,2,1)

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
                       offset,resolution,l_occupied,l_free,l_min,l_max)

# Add the used pose to the trajectory
trajectory = []
trajectory.append(estimatePose)
trajectory.append(estimatePose)

"""
TIME MEASUREMENT
"""

timeLOAD = 0.0
timeFILTER = 0.0
timeMAPPING = 0.0
timeMAPPING_ALL = []
timeMAPPING_ALL.append(0.0)
timeMAPPING_ROUND = []
timeMAPPING_ROUND.append(0.0)
timeMAPPING_BRESEN = []
timeMAPPING_BRESEN.append(0.0)
timeMAPPING_UPDATE = []
timeMAPPING_UPDATE.append(0.0)
timePARTICLE = 0.0
timePARTICLE_ALL = []
timePARTICLE_ALL.append(0.0)
timePARTICLE_OTHER = []
timePARTICLE_OTHER.append(0.0)
timePARTICLE_WEIGHT = []
timePARTICLE_WEIGHT.append(0.0)
timePARTICLE_WEIGHT_TRANSFORM = []
timePARTICLE_WEIGHT_TRANSFORM.append(0.0)
timePARTICLE_WEIGHT_WEIGHT = []
timePARTICLE_WEIGHT_WEIGHT.append(0.0)
timePARTICLE_WEIGHT_SORT = []
timePARTICLE_WEIGHT_SORT.append(0.0)
timeALL = 0.0

for ii in range(1,nrOfRuns):
    t0 = time.time()

    t1 = time.time()    
    pointcloud = np.load('samplePCL.npy')
    timeLOAD = timeLOAD + time.time()-t1
    
    t1 = time.time()
    pointcloud = filterPCL.filterBB(pointcloud,bbPseudoRadius,0.0,0.0)
    pointcloud = filterPCL.filterZ(pointcloud,zCutMin,zCutMax)
    pointcloud = np.delete(pointcloud,2,1)
    timeFILTER = timeFILTER + time.time()-t1
    
    t1 = time.time()
    estimatePose = posEstimation.fitScan2Map2Time(grid,pointcloud,500,1,250,250,
                                                 estimatePose,stddPos,stddYaw,
                                                 startPose,offset,resolution,
                                                 timePARTICLE_ALL,timePARTICLE_OTHER,timePARTICLE_WEIGHT,
                                                 timePARTICLE_WEIGHT_TRANSFORM,timePARTICLE_WEIGHT_WEIGHT,timePARTICLE_WEIGHT_SORT)
    timePARTICLE = timePARTICLE + time.time() - t1
    
    t1 = time.time()
    R = np.matrix([[np.cos(estimatePose[0,2]),-np.sin(estimatePose[0,2])],
                   [np.sin(estimatePose[0,2]),np.cos(estimatePose[0,2])]])
    pointcloud = pointcloud * np.transpose(R)
    pointcloud = pointcloud + np.matrix([estimatePose[0,0],
                                         estimatePose[0,1]])
    mapping.addMeasurementTIME(grid,pointcloud[:,0],pointcloud[:,1],
                           np.matrix([estimatePose[0,0], estimatePose[0,1],0.0]),
                           startPose-offset,resolution,l_occupied,l_free,l_min,l_max,
                           timeMAPPING_ALL,timeMAPPING_BRESEN,timeMAPPING_UPDATE,timeMAPPING_ROUND)
    timeMAPPING = timeMAPPING + time.time() -t1

    trajectory.append(estimatePose)
    timeALL = timeALL + time.time()-t0    
    
    print('Scan '+str(ii)+' processed: '+str(time.time()-t0)+'s')
    
print(pointcloud.shape)

timeALL = timeALL / nrOfRuns
timeFILTER = timeFILTER / nrOfRuns
timeLOAD = timeLOAD / nrOfRuns
timeMAPPING = timeMAPPING / nrOfRuns
timeMAPPING_ALL = timeMAPPING_ALL[-1] / nrOfRuns
timeMAPPING_ROUND = timeMAPPING_ROUND[-1] / nrOfRuns
timeMAPPING_BRESEN = timeMAPPING_BRESEN[-1] / nrOfRuns
timeMAPPING_UPDATE = timeMAPPING_UPDATE[-1] / nrOfRuns
timePARTICLE = timePARTICLE / nrOfRuns
timePARTICLE_ALL = timePARTICLE_ALL[-1] / nrOfRuns
timePARTICLE_WEIGHT = timePARTICLE_WEIGHT[-1] / nrOfRuns
timePARTICLE_OTHER = timePARTICLE_ALL - timePARTICLE_WEIGHT
timePARTICLE_WEIGHT_TRANSFORM = timePARTICLE_WEIGHT_TRANSFORM[-1] / nrOfRuns
timePARTICLE_WEIGHT_WEIGHT = timePARTICLE_WEIGHT_WEIGHT[-1] / nrOfRuns
timePARTICLE_WEIGHT_SORT = timePARTICLE_WEIGHT_SORT[-1] / nrOfRuns

print('Filter: '+str(timeFILTER/timeALL*100)+'%')
print('Load: '+str(timeLOAD/timeALL*100)+'%')
print('Mapping: '+str(timeMAPPING/timeALL*100)+'%')
print('Davon Runden: '+str(timeMAPPING_ROUND/timeMAPPING_ALL*100)+'%')
print('Davon Bresenham: '+str(timeMAPPING_BRESEN/timeMAPPING_ALL*100)+'%')
print('Davon Update: '+str(timeMAPPING_UPDATE/timeMAPPING_ALL*100)+'%')
print('Particle: '+str(timePARTICLE)+',,,%')
print('Davon sonstiges: '+str(timePARTICLE_OTHER)+'...%')
print('Davon weight: '+str(timePARTICLE_WEIGHT)+'....%')
print('DavonDavon transform: '+str(timePARTICLE_WEIGHT_TRANSFORM/timePARTICLE_WEIGHT*100)+'%')
print('DavonDavon weight: '+str(timePARTICLE_WEIGHT_WEIGHT/timePARTICLE_WEIGHT*100)+'%')
print('DavonDavon sort: '+str(timePARTICLE_WEIGHT_SORT/timePARTICLE_WEIGHT*100)+'%')