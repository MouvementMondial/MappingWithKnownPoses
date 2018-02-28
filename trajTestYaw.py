import numpy as np
import matplotlib.pyplot as plt

path = 'C:/KITTI/2011_09_30/2011_09_30_0027_export/'
path = 'D:/KITTI/odometry/dataset/02_export/'

groundTruth = np.asmatrix(np.loadtxt(path+'groundTruth.txt',delimiter=','))
pos = groundTruth

# delete pos from traj
groundTruth = np.delete(groundTruth,0,1)
groundTruth = np.delete(groundTruth,0,1)



groundTruth1 = np.vstack((groundTruth,groundTruth[-1]))
groundTruth2 = np.vstack((groundTruth[0],groundTruth))

delta = np.asarray(groundTruth2-groundTruth1)
delta[delta>=np.pi] -= 2*np.pi
delta[delta<=-np.pi] += 2*np.pi
delta = np.abs(delta)

delta1 = np.vstack((delta,delta[-1]))
delta2 = np.vstack((delta[0],delta))

deltadelta = np.asarray(delta2-delta1)
deltadelta = np.abs(deltadelta)



plt.figure(0)
plt.hist(delta)
print(delta.max())

plt.figure(1)
plt.scatter([pos[:,0]],[pos[:,1]])

plt.figure(2)
plt.hist(deltadelta)
print(deltadelta.max())