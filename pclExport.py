import numpy as np
import pykitti
import utm

from lib import transform

basedir = 'C:/KITTI'
date = '2011_09_26'
drive = '0117'

dataset = pykitti.raw(basedir, date, drive, imformat='cv2')
it = dataset.oxts

# The IMU is mounted at 0.63m height
zCutMin = -0.05
zCutMax = 0.05

"""
Extract Poinctlouds from datastructure, transform them to world coordinate system
and write them to file

Append each position and write the trajectory
"""
traj = []
nr = 0
for scan in dataset.velo:
    print('Process measurement: ' + str(nr))
    
    # get position and orientation angles from trajectory
    pose = next(it)
    lat = pose.packet.lat
    lon = pose.packet.lon
    alt = pose.packet.alt
    roll = pose.packet.roll 
    pitch = pose.packet.pitch
    yaw = pose.packet.yaw
    
    # LaLo to UTM
    pos_cartesian = utm.from_latlon(lat,lon)
    
    # create rotation and translation matrices
    R = transform.rotationMatrix_ypr(yaw,pitch,roll)
    T = np.matrix([pos_cartesian[0], pos_cartesian[1], alt])
        
    # get PCL and delete intensity
    points = np.asarray(scan)
    points = np.delete(points,3,1)

    # transform points
    points = transform.transfromPointcloud(points, R, T)
    
    # save original pointcloud to csv
    np.savetxt('pointcloud_'+str(nr)+'.txt',points,delimiter=',',fmt='%1.3f')

    # get pcl where z-value is in a certain range
    # schneller: vorauswahl der Höhe bereits vor der Transformation und dann 
    # nochmal Feinfilterung der Höhe
    zCutMin_global = zCutMin + alt
    zCutMax_global = zCutMax + alt

    binary = np.logical_and( points[:,2]>zCutMin_global, points[:,2]<zCutMax_global)  
    binary3 = np.column_stack((binary,binary,binary)) 
    pointsFiltered = points[binary3]
    pointsFiltered = np.reshape(pointsFiltered,(-1,3))

    # save filtered poinctloud to csv    
    np.savetxt('pointcloud_zFiltered_'+str(nr)+'.txt',pointsFiltered,delimiter=',',fmt='%1.3f')    
    
    # save traj point
    traj.append(T)
    nr = nr + 1
  
# save trajectory  
traj = np.vstack(traj)
np.savetxt('traj.txt',traj,delimiter=',',fmt='%1.3f')