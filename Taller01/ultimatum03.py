import numpy as np
import matplotlib.pyplot as plt

#Parámetros del dominio
domainX = np.float64(5.0*np.pi)
domainY = np.float64(1.0*np.pi)
#Número de nodos totales
Nx = 31
Ny = 31
#Parámetros de ajuste espacial
kx = 1 
ky = 1
#Tiempo total de la simulación
totalSimTime = 0.02
#Paso temporal
deltaTdim = 0.005
#Coordenadas de posición de las fuentes
x0 = 1
y0 = 1
#Frecuencia angular de oscilación temporal
w = 2
#Parámetros de adimensionalización
tau = np.float64(1/w)
Lc = domainX
Uc = np.float64(Lc/tau)

#Tamaño de la malla
deltaX = np.float64(domainX/(Nx-1))
deltaY = np.float64(domainY/(Ny-1))

#Creo matrices para almacenar las componentes de la velocidad
# en el presente y en el futuro 
uPre = np.ones((Ny,Nx), dtype= np.float64)
vPre = np.ones((Ny,Nx), dtype= np.float64)
uFut = np.ones((Ny,Nx), dtype= np.float64)
vFut = np.ones((Ny,Nx), dtype= np.float64)

#Creo matrices y vectores para almacenar la solución
#Número de incógnitas en el problema (se resta la info de la frontera)
unknownX = np.int64(Nx-2)
unknownY = np.int64(Ny-2)
N = np.int64((unknownX)*(unknownY)) 
A = np.zeros((N, N), dtype= np.float64)
BC1 = np.zeros((N, 1), dtype= np.float64)
BC2 = np.zeros((N, 1), dtype= np.float64)
vectUFut = np.zeros((N, 1), dtype=np.float64)
vectVFut = np.zeros((N, 1), dtype=np.float64)

#Introduzco un conjunto de índices para la malla al estilo Hans Petter
Ix = range(Nx)
Iy = range(Ny)

#Adimensionalizo el tiempo
deltaTnonDim = np.float64(deltaTdim/tau)
nonDimSimTime = np.float64(totalSimTime/tau)
nTsteps = np.int64(nonDimSimTime/deltaTnonDim + 1.0E0)
#Adimensionalizo el espacio
deltaXnonDim = np.float64(deltaX/Lc)
deltaYnonDim = np.float64(deltaY/Lc)

#----------------------------------------------------------------
#Vectores que me servirán para graficar 
nTotalPointsX = np.int64(Nx)
nTotalPointsY = np.int64(Ny)
positionsX = np.zeros(nTotalPointsX, dtype= np.float64)
positionsY = np.zeros(nTotalPointsY, dtype= np.float64)
timeFraction = np.zeros(nTsteps, dtype= np.float64)

for i in range(1,nTotalPointsX):
  positionsX[i] = positionsX[i-1] + deltaX

for i in range(1,nTotalPointsY):
  positionsY[i] = positionsY[i-1] + deltaY

for i in range(1, nTsteps):
  timeFraction[i] = timeFraction[i-1] + deltaTdim

positionsXNonDim = positionsX/Lc
positionsYNonDim = positionsY/Lc
timeFractionNonDim = timeFraction/tau
#----------------------------------------------------------------

#Constantes del problema
k1 = np.float64(Uc*tau/Lc)
k2 = np.float64(tau/Uc)
alfa = np.float64(deltaTnonDim*k1/(4*deltaXnonDim))
beta = np.float64(deltaTnonDim*k1/(4*deltaYnonDim))
gamma = np.float64(deltaTnonDim*k2/2)


#Condiciones iniciales
beginX = np.int64((np.pi)/deltaX + 1)
endX = np.int64((3*np.pi/2)/deltaX + 1)
beginY = np.int64((np.pi/3)/deltaY + 1)
endY = np.int64((2*np.pi/3)/deltaY + 1)
uPre[beginY:endY+1, beginX:endX+1] = 2
vPre[beginY:endY+1, beginX:endX+1] = 2
#Adimensionalizo las condiciones iniciales
uPre = uPre/Uc  
vPre = vPre/Uc
uFut = uFut/Uc  
vFut = vFut/Uc

def uFutCN(Iy, Ix, unknownX, unknownY, alfa, beta, uPre, vPre, uFut):
    N = np.int64((unknownX)*(unknownY)) 
    Ax = np.zeros((N, N), dtype= np.float64)
    Bc1x = np.zeros((N, 1), dtype= np.float64)
    Bc2x = np.zeros((N, 1), dtype= np.float64)
    eqGlobal= 0    #Número de incógnitas globales
    for j in Iy[1:-1]:
        for i in Ix[1:-1]:
            #Línea izquierda
            if  i == 1:
                Bc1x[eqGlobal] = np.float64(-alfa*uPre[j,i]*uFut[j, i-1])
            #Línea inferior
            if j == 1:
                Bc2x[eqGlobal] = np.float64(-beta*vPre[j,i]*uFut[j-1, i])
            #Línea derecha
            if i == unknownX:
                Bc1x[eqGlobal] = np.float64(alfa*uPre[j,i]*uFut[j, i+1])
            #Línea superior
            if j == unknownY:
                Bc2x[eqGlobal] = np.float64(beta*vPre[j,i]*uFut[j+1, i])
            Ax[eqGlobal, eqGlobal] = 1
            if j < unknownY:
                Ax[eqGlobal, eqGlobal+(unknownX)] = beta*vPre[j,i]
            if i < unknownX:
                Ax[eqGlobal, eqGlobal+1] = alfa*uPre[j,i]
            if j > 1:
                Ax[eqGlobal, eqGlobal-(unknownX)] = -beta*vPre[j,i]
            if i > 1:
                Ax[eqGlobal, eqGlobal-1] = -alfa*uPre[j,i]
            eqGlobal += 1

    return Ax, Bc1x, Bc2x

def vFutCN(Iy, Ix, unknownX, unknownY, alfa, beta, uPre, vPre, vFut):
    N = np.int64((unknownX)*(unknownY)) 
    Ay = np.zeros((N, N), dtype= np.float64)
    Bc1y = np.zeros((N, 1), dtype= np.float64)
    Bc2y = np.zeros((N, 1), dtype= np.float64)
    eqGlobal= 0    #Número de incógnitas globales
    for j in Iy[1:-1]:
        for i in Ix[1:-1]:
            #Línea izquierda
            if  i == 1:
                Bc1y[eqGlobal] = np.float64(-alfa*uPre[j,i]*vFut[j, i-1])
            #Línea inferior
            if j == 1:
                Bc2y[eqGlobal] = np.float64(-beta*vPre[j,i]*vFut[j-1, i])
            #Línea derecha
            if i == unknownX:
                Bc1y[eqGlobal] = np.float64(alfa*uPre[j,i]*vFut[j, i+1])
            #Línea superior
            if j == unknownY:
                Bc2y[eqGlobal] = np.float64(beta*vPre[j,i]*vFut[j+1, i])

            Ay[eqGlobal, eqGlobal] = 1
            if j < unknownY:
                Ay[eqGlobal, eqGlobal+(unknownX)] = beta*vPre[j,i]
            if i < unknownX:
                Ay[eqGlobal, eqGlobal+1] = alfa*uPre[j,i]
            if j > 1:
                Ay[eqGlobal, eqGlobal-(unknownX)] = -beta*vPre[j,i]
            if i > 1:
                Ay[eqGlobal, eqGlobal-1] = -alfa*uPre[j,i]
            eqGlobal += 1
    return Ay, Bc1y, Bc2y

def uPreCN(alfa, beta, uPre, vPre, N):
    Vect = np.zeros((N, 1), dtype= np.float64)
    Vect[:] = np.reshape(alfa*uPre[1:-1,1:-1]*uPre[1:-1,:-2] + beta*vPre[1:-1,1:-1]*uPre[:-2, 1:-1] \
        + uPre[1:-1,1:-1] - alfa*uPre[1:-1,1:-1]*uPre[1:-1,2:] - beta*vPre[1:-1,1:-1]*uPre[2:, 1:-1], (N,1))
    return Vect

def vPreCN(alfa, beta, uPre, vPre, N):
    Vect = np.zeros((N, 1), dtype= np.float64)
    Vect[:] = np.reshape(alfa*uPre[1:-1,1:-1]*vPre[1:-1,:-2] + beta*vPre[1:-1,1:-1]*vPre[:-2, 1:-1] \
        + vPre[1:-1,1:-1] - alfa*uPre[1:-1,1:-1]*vPre[1:-1,2:] - beta*vPre[1:-1,1:-1]*vPre[2:, 1:-1], (N,1))
    return Vect

def sourceX(N, w, x0, y0, kx, ky, t, deltaTdim, positionsX, positionsY):
    X, Y = np.meshgrid(positionsX[1:-1], positionsY[1:-1])
    SxPre = np.zeros((N, 1), dtype= np.float64)
    SxFut = np.zeros((N,1), dtype= np.float64)
    SxPre[:] = np.reshape(2*np.exp(-(X-x0)**2/kx-(Y-y0)**2/ky)*np.sin(2*w*timeFraction[t]), (N,1))
    SxFut[:] = np.reshape(2*np.exp(-(X-x0)**2/kx-(Y-y0)**2/ky)*np.sin(2*w*(timeFraction[t]+deltaTdim)), (N,1))
    return SxPre, SxFut

def sourceY(N, w, y0, ky, t, deltaTdim, positionsX, positionsY):
    X, Y = np.meshgrid(positionsX[1:-1], positionsY[1:-1])
    SyPre = np.zeros((N, 1), dtype= np.float64)
    SyFut = np.zeros((N,1), dtype= np.float64)
    SyPre[:] = np.reshape(2*np.exp(-(Y-y0)**2/ky)*np.cos(0.3*w*timeFraction[t]), (N,1))
    SyFut[:] = np.reshape(2*np.exp(-(Y-y0)**2/ky)*np.cos(0.3*w*(timeFraction[t]+deltaTdim)), (N,1))
    return SyPre, SyFut

for t in range(1, nTsteps):
    uFut[1:-1, 1:-1] = -k1*deltaTnonDim/deltaXnonDim*uPre[1:-1,1:-1]*(uPre[1:-1,1:-1]-uPre[1:-1,:-2])\
                    -k1*deltaTnonDim/deltaYnonDim*vPre[1:-1,1:-1]*(uPre[1:-1,1:-1]-uPre[:-2,1:-1])
    
    vFut[1:-1, 1:-1] = -k1*deltaTnonDim/deltaXnonDim*uPre[1:-1,1:-1]*(vPre[1:-1,1:-1]-vPre[1:-1,:-2])\
                -k1*deltaTnonDim/deltaYnonDim*vPre[1:-1,1:-1]*(vPre[1:-1,1:-1]-vPre[:-2,1:-1])
    
    uPre[1:-1,1:-1] = uFut[1:-1,1:-1]
    vPre[1:-1,1:-1] = vFut[1:-1,1:-1]


    if t > 10:
        matAU, Bc1Ualfa, Bc2Ubeta = uFutCN(Iy, Ix, unknownX, unknownY, alfa, beta, uPre, vPre, uFut)
        matAV, Bc1Valfa, Bc2Vbeta = vFutCN(Iy, Ix, unknownX, unknownY, alfa, beta, uPre, vPre, vFut)
        vectUPres = uPreCN(alfa, beta, uPre, vPre, N)
        vectVPres = vPreCN(alfa, beta, uPre, vPre, N)
        SxPre, SxFut = sourceX(N, w, x0, y0, kx, ky, t, deltaTdim, positionsX, positionsY)
        SyPre, SyFut = sourceY(N, w, y0, ky, t, deltaTdim, positionsX, positionsY)

        vectUFut[:] = np.dot(np.linalg.inv(matAU), vectUPres + 0.0*gamma*(SxPre+SxFut) - Bc1Ualfa - Bc2Ubeta)
        vectVFut[:] = np.dot(np.linalg.inv(matAV), vectVPres + 0.0*gamma*(SyPre+SyFut) - Bc1Valfa - Bc2Vbeta)

        uPre[1:-1,1:-1] = np.reshape(vectUFut, (unknownY, unknownX))
        vPre[1:-1,1:-1] = np.reshape(vectVFut, (unknownY, unknownX))
        if t == round(nTsteps*0.25):
            print("25% of simulation done")
        elif t == round(nTsteps*0.50):
            print("50% of simulation done")
        elif t == round(nTsteps*0.75):
            print("75% of simulation done")

X, Y = np.meshgrid(positionsX, positionsY)

fig, ax = plt.subplots(1, 1) 
ax.streamplot(X, Y, uPre, vPre, density = 0.8, color = 'black')
ax.set_title('$Simulation~at~t={:.3f}$'.format(timeFraction[-1]))
ax.set_xlabel('$x[m]$')
ax.set_ylabel('$y[m]$')
ax.set_xlim([0, domainX])
ax.set_ylim([0, domainY])
plt.show()


plt.plot(positionsX, uPre[int(Ny/2)+1, :])
plt.show()