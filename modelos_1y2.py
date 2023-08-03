# -*- coding: utf-8 -*-
"""
Estimación de una funcion de transferencia de primer y segundo orden 

@author: Sebastián Alzate
"""

from control.matlab import *
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import math 

# %%
def graficar(t,y,u,msg):
    #Modifica el tamaño de los ejes
    plt.rcParams.update({'font.size':14})
    
    plt.subplot(111)
    plt.plot(t, y, '-r', linewidth = 2, label = 'Process')
    plt.legend(loc='best')
    plt.grid()
    plt.ylabel('Temperature [°C]')
    plt.xlabel('Time [s]')
    plt.title(msg)
    plt.subplot(111)
    plt.plot(t, u, '-k', linewidth = 2)
    plt.grid()
    #plt.ylabel('input')
    #plt.xlabel('time')
    plt.show()
# %%
data = np.loadtxt('data3.tex',delimiter=',',skiprows=1)

# Extract data columns 
t = data[:,0].T
u = data[:,1].T
y = data[:,2].T


#index = np.where(u>0)
#st = index[0][0]
# Recorte data
#ur = u[st:]
#yr = y[st:]
#tr = t[st:]



# Trasladar los datos 
ut = u 
yt = y - y[1]
tt = t - t[1]



#%% Modelo de la planta primer orden con retardo 

A = np.array([[1, 1/3], [1, 1]])
b = np.array([98.4,265])
x = np.linalg.inv(A) @ b 

# tiempo
Ts = 1
tiem = np.arange(0, 1199, Ts) 

K = 1.04
tau = 250.6279
theta = 14
G = tf(K, [tau, 1])
tdelay = np.arange(0, theta, Ts)

yg, tg, xout= lsim(G, u, tiem)

#plt.figure()
#graficar(tg, yg, u, 'modelo')

# deplazamiento de theta
tg = np.append(tdelay, tg+theta)
yg = np.append(np.ones(len(tdelay))*yg[0],yg)

ug= np.append(np.zeros(len(tdelay))*u[0],u)



plt.figure()
graficar(tt, yt,ut, 'Process vs Model_1')
plt.subplot(1,1,1)
plt.plot(tg, yg, '--g', linewidth = 2, label = 'Frist order')
plt.legend(loc='best')
plt.grid()
# %% Modelo de la planta segundo orden con retardo 
T1 = 55.1
T2 = 166
T3 = 358
X = ((T2-T1)/(T3-T1))
E= ((0.0805 - 5.547*(0.475 - X)**2)/(X - 0.356))
F2 = (2.6*E) - 0.6 
Wn = (F2/(T3 - T1))
F3 = (0.922*(1.66)**E)
THETA = (T2 - (F3/Wn))
K2 = K*Wn
TT = (2*E*math.sqrt(Wn))
G2 = tf(K2, [1, TT, Wn])
y2, t2, xout2= lsim(G2, u, tiem)

dly = np.arange(0, 0, Ts)
t2 = np.append(dly, tg+theta)
y2 = np.append(np.ones(len(dly))*yg[0],yg)
u2= np.append(np.zeros(len(dly))*u[0],u)


plt.figure()
graficar(tt, yt, ut, 'Models')
plt.subplot(1,1,1)
plt.plot(t2, y2, '--b', linewidth = 2, label = 'Second order')
plt.legend(loc='best')
plt.subplot(1,1,1)
plt.plot(tg, yg, '--g', linewidth = 2, label = 'Frist order')
plt.legend(loc='best')

plt.grid()
