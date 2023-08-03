# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:32:09 2022

@author: Sergio
"""
import control
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import *
from controlcae import *


plt.close()

k = 1
TM=14

Gs = tf(1.04, [250.6, 1])
NUM_TM,DEN_TM = pade(TM,1)
TM_TF = control.tf(NUM_TM,DEN_TM)

G_pade=Gs*TM_TF 

H= feedback(G_pade,1)

cG = series(k, H)
#bode(cG)
# Analisis de Estabilidad
gm, pm, Wcg, Wcp = margin(cG)
gm, pm, Wcg, Wcp = margin_plot(cG)
print('Margen de Ganancia:', gm)
print('Margen de Fase:', pm)
print('Frecuencia de Cruce de Fase:', Wcg)
print('Frecuencia de Cruce de Ganancia:', Wcp)

