# -*- coding: utf-8 -*-
"""
Control Proporcional - TCLAB
Función de Transferencia Continua:
                2.34 exp(-8.76s)
    G(s) =  ----------------------
                148.44s + 1
                
Función de Transferencia Discreta:
                 0.1114 z + 0.01138
G(z) = z^(-2) * ---------------------   Ts = 8
                    z - 0.9475
    
@author: Sergio A. Castaño Giraldo
"""
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
import sys
import tclab_cae.tclab_cae as tclab
import time

#Importar funciones modelo NO Lineal
#sys.path.append('../functions') 
#import tclab_fun as fun  


def delay_time(sleep_max, prev_time):
    sleep = sleep_max - (time.time() - prev_time)
    if sleep >= 0.01:
        time.sleep(sleep - 0.01)
    else:
        time.sleep(0.01)
        
    # Record time and change in time
    t = time.time()
    return t
    

plt.close()
# Función de Transferencia Discreta (Profesor)
Ts=30
Gz= tf([0.06447, 0.05313],[1, -0.8869, 0], Ts)    
print(Gz)


#===========================================================#
# Connect to Arduino
lab = tclab.TCLab_CAE()
start_time = time.time()
prev_time = start_time
#sleep time
sleep_max = 1.0
#===========================================================#

# Parametros del Modelo No Lineal
Ta = lab.T1  #lab.T1
Tinit = lab.T1

#Crea los vectores
tsim = 500           #Tiempo de simulacion (segundos)
nit = int(tsim/1)          #Numero de Iteraciones
t = np.zeros(nit)          #Tiempo

#Vectores del proceso Real
u = np.zeros(nit)           #Vector de entrada (Heater)
y = np.zeros(nit)           #Vector de salida  (Temperatura)
e = np.zeros(nit)           #Vector de error


#Vectores del proceso Simulado
us = np.zeros(nit)           #Vector de entrada (Heater)
ys = np.zeros(nit)           #Vector de salida  (Temperatura)
ys[:] = Tinit
es = np.zeros(nit)           #Vector de error

#Setpoint
r = np.zeros(nit)
r[:] = Tinit
r[Ts*2:] = 40

# Control Proporcional
Kc = 4
bias = 0
kss = dcgain(Gz)  #Ganancia del sistema

#Crear plot
plt.figure(figsize=(10,7))
plt.ion() #Enable interactive mode
plt.show()

try:
    #Lazo Cerrado de Control
    for k in range(nit):
        
        tm = delay_time(sleep_max, prev_time)
        prev_time = tm
        t[k] = np.round(tm - start_time)
        
        #=====================================================#
        #============         PROCESO REAL        ============#
        #=====================================================#
        y[k] = lab.T1
        
        #=====================================================#
        #============    SIMULAR EL PROCESO REAL  ============#
        #=====================================================#
        
        #Con el Modelo NO LINEAL
        
        #ys[k] = 0
        
        #Con el Modelo Lineal
        if k > Ts:
            ts = np.arange(0,k+Ts,Ts)
            Tlin, tlin, Xlin= lsim(Gz, us[0:k+Ts:Ts], ts)
            ys[k]=Tlin[-1] + Tinit
        
        #=====================================================#
        #============       CALCULAR EL ERROR     ============#
        #=====================================================#
        e[k]= r[k] - y[k]
        es[k]= r[k] - ys[k]
        
        #=====================================================#
        #===========   CALCULAR LA LEY DE CONTROL  ===========#
        #=====================================================#

        
        if t[k]%Ts == 0:
            u[k] = Kc * e[k] + bias
            us[k] = Kc * es[k] + bias
        else:
            u[k] = u[k-1]
            us[k] = us[k-1]
        
        #Mandar la señal a la planta
        lab.Q1(u[k])
        #SATURACION
        if u[k] > 100: #MAS DE 100 VALE 100
            u[k]=100
        elif u[k] < 0: #MENOS DE 0 VALE 100
            u[k]=0
        
        #SATURACION
        if us[k] > 100: #MAS DE 100 VALE 100
            us[k]=100
        elif us[k] < 0: #MENOS DE 0 VALE 100
            us[k]=0
        
        #Mandar la señal a la planta
        lab.Q1(u[k])
        
        
        #Graficar
        plt.subplot(2,1,1)
        plt.plot(t[0:k],r[0:k],'--k',t[0:k],ys[0:k],'m--',\
                 t[0:k],y[0:k],'r-',linewidth=3)
        plt.legend(['Setpoint', 'Simulation', 'Output'])
        plt.ylabel('Temperature (C)',fontsize=18)
        plt.xlabel('Time(s)',fontsize=18)
        plt.title('Proportional Control',fontsize=24)

        plt.subplot(2,1,2)
        plt.step(t[0:k],us[0:k],'m--',\
                 t[0:k],u[0:k],'b-',linewidth=3)
        plt.legend(['Simulation', 'Heater'])
        plt.ylabel('Power (%)',fontsize=18)
        plt.xlabel('Time(s)',fontsize=18)
        plt.draw()
        plt.pause(0.05)
            
    # Turn off heaters
    lab.Q1(0)
    lab.Q2(0)
    lab.LED(0)
    lab.close()

# Allow user to end loop with Ctrl-C          
except KeyboardInterrupt:
    # Disconnect from Arduino
    lab.Q1(0)
    print('Shutting down')
    lab.close()
       
except:
    # Disconnect from Arduino
    lab.Q1(0)
    lab.Q2(0)
    lab.LED(0)
    lab.close()
    print('Shutting down')
    raise

