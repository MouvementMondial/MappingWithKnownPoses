# -*- coding: utf-8 -*-
"""
@author: Thorsten
"""

import pandas as pd

def readPointcloud2xyz(filename):
    pcl = pd.read_csv(filename, delimiter=',')
    return pcl.as_matrix()
        
def readTraj2xyz(filename):
    traj = pd.read_csv(filename, delimiter=',')
    return traj.as_matrix()
    
