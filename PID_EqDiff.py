import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
import sys
import pandas as pd
from pandas import ExcelWriter


#Importar funciones modelo NO Lineal
#sys.path.append('../functions') 
#import tclab_fun as fun  

from menus import *


plt.close()
Ts   =  30                   #Periodo de Muestreo

#Función de transferencia continua
K = 1.04
theta1 = 14
tau = 250.6
#Corrección del retardo para control discreto
theta = theta1 + Ts/2

# Función de Transferencia Discreta

Gz= tf([0.06447, 0.05313],[1, -0.8869, 0], Ts)  
print(Gz)

# Parametros del Modelo No Lineal
Ta = 28
Tinit = 28

#Crea los vectores
tsim = 700                 #Tiempo de simulacion (segundos)
nit = int(tsim/Ts)          #Numero de Iteraciones
t = np.arange(0, (nit)*Ts, Ts)  #Tiempo
u = np.zeros(nit)           #Vector de entrada (Heater)
y = np.zeros(nit)           #Vector de salida  (Temperatura)
y[:] = Tinit
e = np.zeros(nit)           #Vector de error
q = np.zeros(nit)           #Vector de disturbio
q[40:] = 2

#Setpoint
r = np.zeros(nit)
r[:] = Tinit
r[5:] = 40


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

#Lazo Cerrado de Control
for k in range(nit-1):
    
    #=====================================================#
    #============    SIMULAR EL PROCESO REAL  ============#
    #=====================================================#
    
    #Con el Modelo NO LINEAL
# =============================================================================
#     if k > 1:
#         T1 = fun.temperature_tclab(t[0:k], u[0:k] - q[0:k], Ta, Tinit)
#         y[k] = T1[-1]
# =============================================================================
    
    #Con el Modelo Lineal
    if k > 1:
        Tlin, tlin, Xlin = lsim(Gz, u[0:k+1] - q[0:k+1], t[0:k+1])
        y[k] = Tlin[-1] + Tinit #Agrega la condicion inicial
    
    #=====================================================#
    #============       CALCULAR EL ERROR     ============#
    #=====================================================#
    e[k]= r[k] - y[k]
    
    #=====================================================#
    #===========   CALCULAR LA LEY DE CONTROL  ===========#
    #=====================================================#
    #bias = (y[k]-y[0])/kss
    u[k] = u[k-1] + q0*e[k] + q1*e[k-1] + q2*e[k-2]
    if u[k] > 100:
        u[k] = 100
    elif u[k] < 0:
        u[k] = 0



plt.figure()
plt.subplot(2,1,1)
plt.plot(t,r,'--k',t,y,'r-',linewidth=3)
plt.legend(['Setpoint', 'Output'])
plt.ylabel('Temperature (C)',fontsize=18)
plt.xlabel('Time(s)',fontsize=18)
plt.title('Proportional Control',fontsize=24)

plt.subplot(2,1,2)
plt.step(t,u,'b-',linewidth=3)
plt.legend(['Heater'])
plt.ylabel('Power (%)',fontsize=18)
plt.xlabel('Time(s)',fontsize=18)

df = pd.DataFrame({'t': t,
                   'r': r,
                   'y': y})
writer = ExcelWriter('ejemplo.xlsx')
df.to_excel(writer, 'Hoja de datos', index=False)
writer.save()
