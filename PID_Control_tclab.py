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
    
@author: Sebastián Alzate
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
from menus import *


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
#Función de transferencia continua
Ts   =  30  
K = 1.04
theta1 = 14
tau = 250.6
#Corrección del retardo para control discreto
theta = theta1 + Ts/2

# Función de Transferencia Discreta

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
Ta = lab.T3  #lab.T1
Tinit = lab.T1

#Crea los vectores
tsim = 600                 #Tiempo de simulacion (segundos)
nit = int(tsim/1)          #Numero de Iteraciones
nits = int(tsim/Ts)
t = np.zeros(nit)          #Tiempo

#Vectores del proceso Real
u = np.zeros(nit)           #Vector de entrada (Heater)
y = np.zeros(nit)           #Vector de salida  (Temperatura)
e = np.zeros(nit)           #Vector de error


#Vectores del proceso Simulado
us = np.zeros(nits)           #Vector de entrada (Heater)
ys = np.zeros(nits)           #Vector de salida  (Temperatura)
ys[:] = Tinit
es = np.zeros(nits)           #Vector de error
ts = np.arange(0, (nits)*Ts, Ts)  #Tiempo

#Setpoint
r = np.zeros(nit)
r[:] = Tinit
r[Ts*2:] = 40

#Setpoint Simulado
rs = np.zeros(nits)
rs[:] = Tinit
rs[3:] = 40



if metodo == 1: #Ziegler-Nichols
    if control == 1: #PI
        kp=(0.9*tau)/(K*theta)*0.7
        ti=theta/0.3
        td=0
        incontrolabilidad = theta / tau
    else: #PID
        kp=(1.2*tau)/(K*theta)*0.7
        ti=2*theta
        td=0.5*theta
        incontrolabilidad = theta / tau



if metodo == 2: #IAE
    if control == 1: #PI
        kp=(1/K)*(0.984*(theta/tau)**(-0.986))
        ti=(tau)/(0.608*(theta/tau)**(-0.707))
        td=0
        incontrolabilidad = theta / tau
    else: #PID
        kp=(1/K)*(1.435*(theta/tau)**(-0.921))
        ti=(tau)/(0.878*(theta/tau)**(-0.749))
        td=(tau)*(0.482*(theta/tau)**(1.137))
        incontrolabilidad = theta / tau
    


if metodo == 3: #IAET
    if control == 1: #PI
        kp=(1/K)*(0.849*(theta/tau)**-0.977)
        ti=(tau)/(0.674*(theta/tau)**-0.68)
        td=0
        incontrolabilidad = theta / tau
    else: #PID
        kp=(1/K)*(1.357*(theta/tau)**-0.947)*0.7
        ti=(tau)/(0.842*(theta/tau)**-0.738)
        td=(tau)*(0.381*(theta/tau)**0.995)
        incontrolabilidad = theta / tau

if metodo == 4: #Cohen - Coon
    if control == 1: #PI
        kp=(0.9+((0.083)*(theta/tau)))*((tau)/(K*theta))
        ti=(theta*(0.9+((0.083)*(theta/tau))))/((1.27+((0.6)*(theta/tau))))
        td=0
        incontrolabilidad = theta / tau
    else: #PID
        kp=(1.35+((0.25)*(theta/tau)))*((tau)/(K*theta))*0.7
        ti=(theta*(1.35+((0.25)*(theta/tau))))/((0.54+((0.33)*(theta/tau))))
        td=(0.5*theta)/(1.35+((0.25)*(theta/tau)))
        incontrolabilidad = theta / tau
    
if metodo == 5: #Asignación de polos
    if control == 1: #PI
        kp=6.24
        ti=43.16
        td=0
        incontrolabilidad = theta / tau
    else: #PID
        kp=6.24
        ti=43.16
        td=0
        incontrolabilidad = theta / tau


#Calculo do controle PID digital
q0=kp*(1+Ts/(2*ti)+td/Ts)
q1=-kp*(1-Ts/(2*ti)+(2*td)/Ts)
q2=(kp*td)/Ts

#Crear plot
plt.figure(figsize=(10,7))
plt.ion() #Enable interactive mode
plt.show()

ks = 0 #Instante k simulado
try:
    #Lazo Cerrado de Control
    for k in range(nit- 1):
        
        tm = delay_time(sleep_max, prev_time)
        prev_time = tm
        t[k] = np.round(tm - start_time) - 1
        
        #=====================================================#
        #============         PROCESO REAL        ============#
        #=====================================================#
        y[k] = lab.T1
        
        #=====================================================#
        #============    SIMULAR EL PROCESO REAL  ============#
        #=====================================================#
        
        #Con el Modelo NO LINEAL
# =============================================================================
#         if k > 1:
#             T1 = fun.temperature_tclab(t[0:k], u[0:k] - q[0:k], Ta, Tinit)
#             y[k] = T1[-1]
# =============================================================================
        
        #Con el Modelo Lineal
        if k > Ts and t[k]%Ts == 0:
# =============================================================================
#             ts = np.arange(0,len(us[0:k+Ts:Ts])*Ts, Ts)
#             Tlin, tlin, Xlin = lsim(Gz, us[0:k+Ts:Ts], ts)
#             ys[k] = Tlin[-1] + Tinit #Agrega la condicion inicial
# =============================================================================
            
            Tlin, tlin, Xlin = lsim(Gz, us[0:ks+1], ts[0:ks+1])
            ys[ks] = Tlin[-1] + Tinit #Agrega la condicion inicial
            es[ks]= rs[ks] - ys[ks] # Error simulado
            
            
        
        
        #=====================================================#
        #============       CALCULAR EL ERROR     ============#
        #=====================================================#
        e[k]= r[k] - y[k]
        #es[k]= r[k] - ys[k]
        
        #=====================================================#
        #===========   CALCULAR LA LEY DE CONTROL  ===========#
        #=====================================================#
        #bias = (y[k]-y[0])/kss
        if t[k]%Ts == 0 and k > Ts:
            u[k] =  u[k-1*Ts] + q0*e[k] + q1*e[k-1*Ts] + q2*e[k-2*Ts]
            us[ks] = us[ks-1] + q0*es[ks] + q1*es[ks-1] + q2*es[ks-2]
        else:
            u[k] = u[k-1]
            #us[k] = us[k-1]
            
        if u[k] > 100:
            u[k] = 100
        elif u[k] < 0:
            u[k] = 0
        
        #Saturación simulado
        if us[ks] > 100:
            us[ks] = 100
        elif us[ks] < 0:
            us[ks] = 0
            
        if t[k]%Ts == 0:
            ks += 1
        
        # write Heater (0 -100)
        lab.Q1(u[k])
        
        #Graficar
        plt.subplot(2,1,1)
        plt.plot(t[0:k], r[0:k],'--k',linewidth=3)
        #plt.hold
        plt.plot(ts[0:ks], ys[0:ks],'m--',linewidth=3)
        plt.plot(t[0:k], y[0:k],'r-',linewidth=3)
        plt.legend(['Setpoint', 'Simulation', 'Output'])
        plt.ylabel('Temperature (C)',fontsize=18)
        plt.xlabel('Time(s)',fontsize=18)
        plt.title('Proportional Control',fontsize=24)

        plt.subplot(2,1,2)
        plt.step(ts[0:ks],us[0:ks],'m--', linewidth=3)
        plt.step(t[0:k],u[0:k],'b-',linewidth=3)
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

