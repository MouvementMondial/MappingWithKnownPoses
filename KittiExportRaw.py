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
    np.savetxt(pathSave+'pointcloud_'+str(nr)+'.txt',pointcloud,delimiter=',',fmt='%1.3f')
    #np.save(pathSave+'pointcloudNP_'+str(nr),pointcloud)
    """
    Extract ground truth
    """
    pose = next(gpsImu)
    latitude = pose.packet.lat
    longitude = pose.packet.lon
    # latitude and longitude to UTM
    posUTM = utm.from_latlon(latitude,longitude)
    # add position x y yaw to list
    groundTruth.append(np.matrix([posUTM[0], posUTM[1], pose.packet.yaw]))
    
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

"""
Extract and save calibration (transformation matrix GPS/IMU -> Velodyne)
"""

R_Imu2Velod = np.matrix(dataset.calib.T_velo_imu[0:3,0:3])
T_Imu2Velod = np.matrix(dataset.calib.T_velo_imu[0:3,3]).transpose()
trans_Imu2Velod = np.hstack((R_Imu2Velod,T_Imu2Velod))
np.savetxt(pathSave+'calib.txt',trans_Imu2Velod,delimiter=',',fmt='%1.8f')
