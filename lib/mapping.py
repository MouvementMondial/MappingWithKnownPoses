# -*- coding: utf-8 -*-
"""
@author: Thorsten
"""

from lib import bresenham
import numpy as np
from numba import jit

@jit
def addMeasurement(grid, x, y, pos_sensor, offset, resolution, l_occupied, l_free, l_min, l_max):
    
    for i in range(x.size):
        # round points to cells 
        xi=int( (x[i,0]-offset[0,0]) / resolution )
        yi=int( (y[i,0]-offset[0,1]) / resolution )

        # set beam endpoint-cells as occupied
        grid[xi,yi] += l_occupied
    
        # value > threshold? -> clamping
        #if grid[xi,yi] > l_max:
        #    grid[xi,yi] = l_max
        grid[xi,yi] = min((grid[xi, yi], l_max))

        '''
        Es ist ganz anschaulich eine oder beide nächsten Befehle aus-
        zukommentieren. Gerade letzterer ist sehr teuer
        '''
        # calculate cells between sensor and endpoint as free
        path = bresenham.bresenham2D( ((pos_sensor-offset)/resolution).astype(int), np.array([[xi,yi]]))
        
        # set cells between sensor and endpoint as free
        updateFree(path,grid,l_free,l_min)
        
'''
Habe folgendes in eine eigene Funktion geschrieben, da ich die Hoffnung hatte,
dass man es mit numba im nopython-Mode kompilieren kann (allokieren von Arrays
ist angeblich nicht möglich), aber es funktioniert nicht.
So wird vermutlich  noch über den object-mode auf python zugegriffen, sodass kaum
Verbesserung zu erwarten ist.

http://numba.pydata.org/numba-doc/0.17.0/glossary.html#term-object-mode

@jit(nopython=True)
'''  
              
@jit
def updateFree(path,grid,l_free,l_min):
    
    for nr in range(path.shape[0]):
        
        # Indizes direkt als neue Variablen speichern reduziert den Aufwand erheblich!
        # Im Extremfall würde ich die Berechnung der Indizes sogar vor die Schleife auslagern.
        path_x = int(path[nr,0])
        path_y = int(path[nr,1])
        
        grid[path_x, path_y] += l_free
        
        # value < threshold? -> clamping        
        if grid[path_x, path_y] < l_min:
            grid[path_x, path_y] = l_min
            
        """    
        grid[int(path[nr,0]),int(path[nr,1])] += l_free
        
        # value < threshold? -> clamping        
        if grid[int(path[nr,0]),int(path[nr,1])] < l_min:
            grid[int(path[nr,0]),int(path[nr,1])] = l_min
        """