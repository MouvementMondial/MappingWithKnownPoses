# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:16:45 2018

@author: Thorsten
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

font = {'size'   : 20}
plt.rc('font', **font)

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

path = 'D:/KITTI/odometry/dataset/07_export/'

nr = 10

groundTruth = np.asmatrix(np.loadtxt(path+'groundTruth.txt',delimiter=','))
ax = plt.subplot(111) 
plt.axis('equal')
plt.scatter([groundTruth[:,1]],[groundTruth[:,0]],
            c='g',s=20,edgecolors='none', label = 'Trajektorie Ground Truth')




trajs = np.asmatrix(np.loadtxt(path+'07_trajs_slatchOHNE_01.txt',delimiter=','))
# mean trajs
meanX = np.mean(trajs[:,::2],axis=1)
meanY = np.mean(trajs[:,1::2],axis=1)
covSum2 = []
for ii in range(0,trajs.shape[0],10):
    cov = np.cov(trajs[ii,1::2].tolist(),trajs[ii,::2].tolist())
    covSum2.append(np.trace(cov))
    nstd = 2
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(trajs[ii,1::2]),np.mean(trajs[ii,::2])),
              width=w, height=h,
              angle=theta, color='blue')
    ell.set_facecolor('none')
    ax.add_artist(ell)
for ii in range(0,nr*2-1,2):
    plt.scatter([trajs[::10,ii+1]],[trajs[::10,ii]],
                c='r',s=3,edgecolors='none')
plt.scatter([meanY],[meanX],c='b',s=20,edgecolors='none',label = 'Trajektorien 1.000 Partikel')

trajectory = np.hstack((meanX,meanY))
errorPos2 = np.sqrt( np.multiply(trajectory[:,0]-groundTruth[:,0],trajectory[:,0]-groundTruth[:,0])
                     +np.multiply(trajectory[:,1]-groundTruth[:,1],trajectory[:,1]-groundTruth[:,1]))

trajs = np.asmatrix(np.loadtxt(path+'07_trajs_slatchOHNE_01_10000.txt',delimiter=','))
# mean trajs
meanX = np.mean(trajs[:,::2],axis=1)
meanY = np.mean(trajs[:,1::2],axis=1)
covSum1 = []
for ii in range(0,trajs.shape[0],10):
    cov = np.cov(trajs[ii,1::2].tolist(),trajs[ii,::2].tolist())
    covSum1.append(np.trace(cov))    
    nstd = 2
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(trajs[ii,1::2]),np.mean(trajs[ii,::2])),
              width=w, height=h,
              angle=theta, color='red')
    ell.set_facecolor('none')
    ax.add_artist(ell)
for ii in range(0,nr*2-1,2):
    if ii == 0:
        plt.scatter([trajs[::10,ii+1]],[trajs[::10,ii]],
                    c='r',s=3,edgecolors='none')
    else:
        plt.scatter([trajs[::10,ii+1]],[trajs[::10,ii]],
                    c='r',s=3,edgecolors='none')
plt.scatter([meanY],[meanX],c='r',s=20,edgecolors='none',label = 'Trajektorien 10.000 Partikel')

trajectory = np.hstack((meanX,meanY))
errorPos1 = np.sqrt( np.multiply(trajectory[:,0]-groundTruth[:,0],trajectory[:,0]-groundTruth[:,0])
                     +np.multiply(trajectory[:,1]-groundTruth[:,1],trajectory[:,1]-groundTruth[:,1]))

trajs = np.asmatrix(np.loadtxt(path+'07_trajs_slatch_01_500_250_250.txt',delimiter=','))
# mean trajs
meanX = np.mean(trajs[:,::2],axis=1)
meanY = np.mean(trajs[:,1::2],axis=1)
covSum2 = []
for ii in range(0,trajs.shape[0],10):
    cov = np.cov(trajs[ii,1::2].tolist(),trajs[ii,::2].tolist())
    covSum2.append(np.trace(cov))
    nstd = 2
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(trajs[ii,1::2]),np.mean(trajs[ii,::2])),
              width=w, height=h,
              angle=theta, color='magenta')
    ell.set_facecolor('none')
    ax.add_artist(ell)
for ii in range(0,nr*2-1,2):
    plt.scatter([trajs[::10,ii+1]],[trajs[::10,ii]],
                c='m',s=3,edgecolors='none')
plt.scatter([meanY],[meanX],c='m',s=20,edgecolors='none',label = 'Trajektorien 500 + 250 + 250 Partikel')

trajectory = np.hstack((meanX,meanY))
errorPos2 = np.sqrt( np.multiply(trajectory[:,0]-groundTruth[:,0],trajectory[:,0]-groundTruth[:,0])
                     +np.multiply(trajectory[:,1]-groundTruth[:,1],trajectory[:,1]-groundTruth[:,1]))

plt.legend()







# traveled distance ground truth
temp = np.vstack((groundTruth[0,:],groundTruth))
temp = np.delete(temp,(temp.shape[0]-1), axis=0)
distanceGT = np.sqrt( np.multiply(temp[:,0]-groundTruth[:,0],temp[:,0]-groundTruth[:,0])
                     +np.multiply(temp[:,1]-groundTruth[:,1],temp[:,1]-groundTruth[:,1]))
distanceGT = np.transpose(np.cumsum(distanceGT))

plt.figure(2)
plt.title('Entwicklung Varianz')
plt.plot(distanceGT[::10,:],covSum1,c='b', label = 'Ohne Filterung')
plt.plot(distanceGT[::10,:],covSum2,c='m', label = 'Mit Filterung')
plt.xlabel('Distanz')
plt.ylabel('Varianz')
plt.legend()

plt.figure(3)


plt.plot(distanceGT,errorPos1,c='b', label = 'Ohne Filterung',linewidth=2)
plt.plot(distanceGT,errorPos2,c='m', label = 'Mit Filterung',linewidth=2)
plt.xlabel('Distanz Ground Truth [m]')
plt.ylabel('Abweichung [m]')
plt.grid()
plt.legend(loc='upper left')


"""
xx = trajs[269,::2]
yy = trajs[269,1::2]
plt.figure(2)
plt.subplot(311)
plt.hist(np.transpose(xx))
plt.subplot(312)
plt.hist(np.transpose(yy))
plt.subplot(313)
plt.hist2d(np.asarray(xx)[:,0],np.asarray(yy)[:,0])
"""
