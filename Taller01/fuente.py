import numpy as np
import matplotlib.pyplot as plt

#Parámetros del dominio
domainX = np.float64(5.0*np.pi)
domainY = np.float64(1.0*np.pi)
#Número de nodos totales
Nx = 61
Ny = 31
#Parámetros de ajuste espacial
kx = 1 
ky = 1
#Tiempo total de la simulación
totalSimTime = 20
#Paso temporal
deltaTdim = 0.1
#Coordenadas de posición de las fuentes
x0 = 4
y0 = 6
#Frecuencia angular de oscilación temporal
w = 1
#Parámetros de adimensionalización
tau = np.float64(1/w)
Lc = domainX
Uc = np.float64(Lc/tau)

#Tamaño de la malla
deltaX = np.float64(domainX/(Nx-1))
deltaY = np.float64(domainY/(Ny-1))

#Adimensionalizo el tiempo
deltaTnonDim = np.float64(deltaTdim/tau)
nonDimSimTime = np.float64(totalSimTime/tau)
nTsteps = np.int64(nonDimSimTime/deltaTnonDim + 1.0E0)

nTotalPointsX = np.int64(Nx)
nTotalPointsY = np.int64(Ny)
positionsX = np.zeros(nTotalPointsX, dtype= np.float64)
positionsY = np.zeros(nTotalPointsY, dtype= np.float64)
timeFraction = np.zeros(nTsteps, dtype= np.float64)

unknownX = np.int64(Nx-2)
unknownY = np.int64(Ny-2)
N = np.int64((unknownX)*(unknownY)) 

for i in range(1,nTotalPointsX):
  positionsX[i] = positionsX[i-1] + deltaX

for i in range(1,nTotalPointsY):
  positionsY[i] = positionsY[i-1] + deltaY

for i in range(1, nTsteps):
  timeFraction[i] = timeFraction[i-1] + deltaTdim

def sourceX(unknownX, unknownY, w, x0, y0, kx, ky, t, deltaTdim, positionsX, positionsY):
    X, Y = np.meshgrid(positionsX[1:-1], positionsY[1:-1])
    SxPre = np.zeros((unknownY, unknownX), dtype= np.float64)
    SxFut = np.zeros((unknownY,unknownX), dtype= np.float64)
    SxPre[:, :] = 2*np.exp(-(X-x0)**2/kx-(Y-y0)**2/ky)*np.sin(2*w*timeFraction[t])
    SxFut[:, :] = 2*np.exp(-(X-x0)**2/kx-(Y-y0)**2/ky)*np.sin(2*w*(timeFraction[t]+deltaTdim))
    return SxPre, SxFut, X, Y

def sourceY(unknownX, unknownY, w, y0, ky, t, deltaTdim, positionsX, positionsY):
    X, Y = np.meshgrid(positionsX[1:-1], positionsY[1:-1])
    SyPre = np.zeros((unknownY, unknownX), dtype= np.float64)
    SyFut = np.zeros((unknownY, unknownX), dtype= np.float64)
    SyPre[:, :] = 2*np.exp(-(Y-y0)**2/ky)*np.cos(0.3*w*timeFraction[t])
    SyFut[:, :] = 2*np.exp(-(Y-y0)**2/ky)*np.cos(0.3*w*(timeFraction[t]+deltaTdim))
    return SyPre, SyFut, X, Y

SxPre, SxFut, X1, Y1 = sourceX(unknownX, unknownY, w, x0, y0, kx, ky, 40, deltaTdim, positionsX, positionsY)
SyPre, SyFut, X2, Y2 = sourceY(unknownX, unknownY, w, y0, ky, 10, deltaTdim, positionsX, positionsY)

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(X1, Y1, SxPre, cmap='inferno', alpha=0.8)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('z', fontsize=12)
 
plt.show()

fig2 = plt.figure(figsize=(10, 8))
ax2 = plt.axes(projection='3d')
ax2.plot_surface(X2, Y2, SyPre, cmap='inferno', alpha=0.8)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_zlabel('z', fontsize=12)
 
plt.show()