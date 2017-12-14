# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:09:47 2017

@author: Thorsten

based on particle filter demo by Martin J. Laubach
https://github.com/mjl/particle_filter_demo

"""

class Particle(object):
    
    def __init__(self, x, y, yaw, weight):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.weight = weight