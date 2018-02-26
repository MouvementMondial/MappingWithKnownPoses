# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:16:02 2018

@author: Thorsten
"""
import numpy as np
import matplotlib.pyplot as plt

font = {'size'   : 20}
plt.rc('font', **font)


labels = 'Mapping', 'Lokalisierung','Sonstiges'
fracs = [53.6,44.5,1.9]
explode=( 0.1, 0.1, 0.1)
plt.axis("equal")
plt.pie(fracs,explode=explode,labels=labels,shadow=True,autopct="%1.1f%%",startangle=45)
plt.title('Prozentuale Rechenzeit SLAM')