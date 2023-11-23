import numpy as np
import matplotlib.pyplot as plt

Nx = 4
Ny = 4

u = np.zeros((Nx+1,Ny+1), dtype= np.float32)
Bc = np.zeros((Nx-1,1), dtype= np.float32)

Ix = range(Nx+1)
Iy = range(Ny+1)

gamma = 3
alfa = 4
beta = 2

N = (Nx-1)*(Ny-1)
A = np.zeros((N, N))
Bc1 = np.zeros((N, 1))
Bc2 = np.zeros((N, 1))


eqGlobal= 0    #Númeor de incógnitas globales
eqLocal = Nx-1 #Número de incógnitas por fila en x
for i in Ix[1:-1]:
    for j in Iy[1:-1]:
        #Línea izquierda
        if  i == 1:
            Bc1[eqGlobal] = beta  
        #Línea inferior
        if j == 1:
            Bc2[eqGlobal] = alfa
        #Línea derecha
        if i == Nx-1:
            Bc1[eqGlobal] = beta 
        #Línea superior
        if j == Ny-1:
            Bc2[eqGlobal] = alfa

        A[eqGlobal, eqGlobal] = -gamma

        if i < Nx-1:
            A[eqGlobal, eqGlobal+(Ny-1)] = beta
        if j < Ny-1:
            A[eqGlobal, eqGlobal+1] = alfa
        if j > 1:
            A[eqGlobal, eqGlobal-1] = alfa
        if i > 1:
            A[eqGlobal, eqGlobal-(Ny-1)] = beta
        eqGlobal += 1

print(Bc1)
print(Bc2)
print(A)

