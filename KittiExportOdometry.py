# -*- coding: utf-8 -*-
"""
Load KITTI odometry data with pykitti and export the data for SLAM.

https://github.com/utiasSTARS/pykitti

pointclouds:
The pointclouds are saved in the velodyne coordinate system as txt: x y z

start pose:
The start pose is saved in the GPS/IMU coordinate system (UTM) as txt: x y yaw

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
import os

"""
Load kitti dataset
"""
# odometry
basedir = 'D:/KITTI/odometry/dataset'
sequence = '20'
pathSave = basedir+'/'+sequence+'_export/'
if not os.path.exists(pathSave):
    os.makedirs(pathSave)
    
dataset = pykitti.odometry(basedir, sequence)

"""
Extract and save calibration (transformation matrix GPS/IMU -> Velodyne)
"""
R_Cam2Velod = np.matrix(dataset.calib.T_cam0_velo[0:3,0:3])
T_Cam2Velod = np.matrix(dataset.calib.T_cam0_velo[0:3,3]).transpose()

"""
Export ground truth from poses
"""
"""
transMatrices = np.loadtxt(basedir+'/poses/'+sequence+'.txt',delimiter=' ')
trajectory = np.delete(transMatrices,[0,1,2,4,5,6,8,9,10],1)
#trajectory = trajectory + np.transpose(T_Cam2Velod)
trajectory = np.dot(np.transpose(R_Cam2Velod),np.transpose(trajectory))
#offset = np.matrix([[-T_Cam2Velod[2,0]],[T_Cam2Velod[0,0]],[T_Cam2Velod[1,0]]])
#trajectory = trajectory + offset
groundTruth = np.transpose(trajectory)
"""
"""
Process all measurements
"""
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
    np.save(pathSave+'pointcloudNP_'+str(nr),pointcloud)
    
    nr = nr + 1
"""
Save first pose
"""    
np.savetxt(pathSave+'firstPose.txt',np.matrix(([0.0,0.0,0.0])),delimiter=',',fmt='%1.3f')

"""    
Save ground truth
"""
#np.savetxt(pathSave+'groundTruth.txt',groundTruth,delimiter=',',fmt='%1.3f')

