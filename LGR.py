# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:31:17 2022

@author: Usuario
"""

#polos y ceros
import control
import math
from control.matlab import *
import numpy as np
import matplotlib.pyplot as plt


#TIEMPO CONTINUO



#TIEMPO discreto
Ts=30
Gz= tf([0.06447, 0.05313],[1, -0.8869, 0], Ts)

#-----------------------
TM=14

Gs = tf(1.04, [250.6, 1])
NUM_TM,DEN_TM = pade(TM,1)
TM_TF = control.tf(NUM_TM,DEN_TM)

G_pade=Gs*TM_TF 

H= feedback(G_pade,1)

rlocus(H)
##POLOS Y CEROS
'''
plt.figure()
plt.subplot(1,2,1)
pzmap(Gs,grid=True)

plt.subplot(1,2,2)
pzmap(Gz,grid=True)



#RESPUESTA DINAMICA
y,t = step(Gz)
plt.figure(3)
plt.plot(t,y)

#POLOS Y CEROS
#continuo
print('='*30)
print('Polos continuos:')
i=1
for p in pole(Gs):
    print(f'Polo {i} = {p}')
    i+=1
i=1 
print('Zeros continuos:')  
for p in zeros(Gs):
    print(f'Polo {i} = {p}')
    i+=1
    
#discreto
print('='*30)
print('Polos discretos:')
i=1
for p in pole(Gz):
    print(f'Polo {i} = {p}')
    i+=1
i=1 
print('Zeros discretos:')  
for p in zeros(Gz):
    print(f'Cero {i} = {p}')
    i+=1
'''




