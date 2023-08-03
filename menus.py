# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 12:10:04 2022

@author: Sergio
"""

menu1 = """ 
Seleccione el método de ajuste del controlador:
    [1] Ziegler - Nichols
    [2] IAE
    [3] IAET
    [4] Cohen - Coon
    [5] Asignación de Polos
Seleccione una opción entre 1-5: """

menu2 = """Seleccione el método de ajuste del controlador:
    [1] PI
    [2] PID
Seleccione una opción entre 1-2: """


metodo = int(input(menu1))
while not (0 < metodo < 6):
    print("\n Error, escoja un valor válido")
    metodo = int(input(menu1))
    
control = int(input(menu2))
while not (0 < metodo < 6):
    print("\n Error, escoja un valor válido")
    metodo = int(input(menu2))
