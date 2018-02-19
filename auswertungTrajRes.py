import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

font = {'size'   : 20}
plt.rc('font', **font)


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

path = 'C:/KITTI/2011_09_30/2011_09_30_0027_export/'

nr = 25

groundTruth = np.asmatrix(np.loadtxt(path+'groundTruth.txt',delimiter=','))

# traveled distance ground truth
temp = np.vstack((groundTruth[0,:],groundTruth))
temp = np.delete(temp,(temp.shape[0]-1), axis=0)
distanceGT = np.sqrt( np.multiply(temp[:,0]-groundTruth[:,0],temp[:,0]-groundTruth[:,0])
                     +np.multiply(temp[:,1]-groundTruth[:,1],temp[:,1]-groundTruth[:,1]))
distanceGT = np.transpose(np.cumsum(distanceGT))

            
filenamesTrajs = []
filenamesTrajs.append('0.1')
filenamesTrajs.append('0.15')
filenamesTrajs.append('0.2')
filenamesTrajs.append('0.25')
filenamesTrajs.append('0.3')
filenamesTrajs.append('0.35')
filenamesTrajs.append('0.4')
filenamesTrajs.append('0.45')
for res in filenamesTrajs:
    trajs = np.asmatrix(np.loadtxt(path+'trajs_'+res+'.txt',delimiter=','))

    meanX = np.mean(trajs[:,::2],axis=1)
    meanY = np.mean(trajs[:,1::2],axis=1)

    trajectory = np.hstack((meanX,meanY))
    errorPos = np.sqrt( np.multiply(trajectory[:,0]-groundTruth[:,0],trajectory[:,0]-groundTruth[:,0])
                        +np.multiply(trajectory[:,1]-groundTruth[:,1],trajectory[:,1]-groundTruth[:,1]))    

    plt.plot(distanceGT,errorPos,label='Zellgröße: '+res+'m',linewidth=2)    
    
    """
    for ii in range(0,nr*2-1,2):
        
        trajectory = np.hstack((trajs[:,ii],trajs[:,ii+1]))
        errorPos = np.sqrt( np.multiply(trajectory[:,0]-groundTruth[:,0],trajectory[:,0]-groundTruth[:,0])
                            +np.multiply(trajectory[:,1]-groundTruth[:,1],trajectory[:,1]-groundTruth[:,1]))
        if(ii==0):
            plt.plot(distanceGT,errorPos,label=res)
        else:
            plt.plot(distanceGT,errorPos)
    """            
plt.legend(loc='upper left')
plt.title('Abweichung bei verschiedenen Zellgrößen')
plt.xlabel('Distanz GT [m]')
plt.ylabel('Abweichung [m]')