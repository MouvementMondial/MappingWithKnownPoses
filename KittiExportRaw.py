# -*- coding: utf-8 -*-
"""
Load KITTI raw data with pykitti and export the data for SLAM.

https://github.com/utiasSTARS/pykitti

pointclouds:
The pointclouds are saved in the velodyne coordinate system as txt: x y z

start pose:
The start pose is saved in the GPS/IMU coordinate system (UTM) as txt: x y yaw

ground truth:
The ground truth is a list of poses in the GPS/IMU coordinate system (UTM), 
saved as txt: x y yaw

calib: 3x4 Transformation matrix (3x3 rotation matrix | 3x1 translation vector),
transform a point from GPS/IMU coordinate system to velodyne coordinate system,
saved as txt.

All coordinate systems as described in the KITTI readme. (see raw data 
development kit)

Please download the data and the raw data development kit:
http://www.cvlibs.net/datasets/kitti/raw_data.php

@author: Thorsten
"""

import numpy as np
import pykitti
import utm
import os

"""
Load kitti dataset
"""
basedir = 'C:/KITTI'
date = '2011_09_30'
drive = '0027'
pathSave = basedir+'/'+date+'/'+date+'_'+drive+'_export/'
if not os.path.exists(pathSave):
    os.makedirs(pathSave)

dataset = pykitti.raw(basedir,date,drive)
gpsImu = dataset.oxts

"""
Extract calibration (translation GPS/IMU -> Velodyne)
"""
T = np.matrix(dataset.calib.T_velo_imu[0:2,3]).transpose()

"""
Process all measurements
"""
groundTruth = []
nr = 0

for scan in dataset.velo:
    print('Process measurement: '+str(nr))
    
    """
    Extract and save pointcloud
    """
    # get pointcloud (x y z intensity) and delete intensity
    pointcloud = np.asarray(scan)
    pointcloud = np.delete(pointcloud,3,1)
    
    # save pointcloud as txt
    #np.savetxt(pathSave+'pointcloud_'+str(nr)+'.txt',pointcloud,delimiter=',',fmt='%1.3f')
    
    # save pointcloud as binary    
    np.save(pathSave+'pointcloudNP_'+str(nr),pointcloud)

    """
    Extract ground truth and transform it to velodyne position
    """
    pose = next(gpsImu)
    latitude = pose.packet.lat
    longitude = pose.packet.lon
    yaw = pose.packet.yaw
    # latitude and longitude to UTM
    posUtm = utm.from_latlon(latitude,longitude)
    posUtm = np.matrix([posUtm[0],posUtm[1]])
    # Rotation Matrix to rotate IMU2VELO translation vector
    R = np.matrix([[np.cos(yaw), -np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    
    # Rotate translation vector 
    T_rot = R*T        

    # Calculate Ground Truth at the position of the velodyne LIDAR
    posUtmTrans = posUtm-np.transpose(T_rot)
    
    # add position x y yaw to list
    groundTruth.append(np.matrix([posUtmTrans[0,0], posUtmTrans[0,1], pose.packet.yaw]))
    
    nr = nr + 1

"""
Save first pose
"""    
np.savetxt(pathSave+'firstPose.txt',groundTruth[0],delimiter=',',fmt='%1.3f')

"""    
Save ground truth
"""
groundTruth = np.vstack(groundTruth)
np.savetxt(pathSave+'groundTruth.txt',groundTruth,delimiter=',',fmt='%1.3f')


