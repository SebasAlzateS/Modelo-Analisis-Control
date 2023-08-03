# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 11:47:15 2022

@author: Usuario
"""

import control
import math
from control.matlab import *
import numpy as np
import matplotlib.pyplot as plt

Gs = tf([1.04], [250.6, 1])
theta=14

H= feedback(Gs,1)
H_norm= tf([1.04/2.04], [250.6/2.04, 2.04/2.04])
tau=122.8
r1=0.2*(tau+theta)
r2=0.6*(tau+theta)
print(r1,"<= T <=",r2)