import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------
#Con esta función calculo coordenadas en una dimensión para los nodos
def facesCoordinates(xPoints, yPoints, xDimen, yDimen):
    xDelta = np.float64(xDimen/(xPoints))
    yDelta = np.float64(yDimen/(yPoints))

    #Defino vectores auxiliares dentro de la función porque no son tan útiles
    xCoordSN = np.zeros((xPoints,), dtype = np.dtype("float64"))
    yCoordWE = np.zeros((yPoints,), dtype = np.dtype("float64"))

    #Para las coordenadas de las caras norte y sur
    yCoordSN = np.linspace(0.0, yDimen, yPoints+1, endpoint = True, dtype = np.dtype("float64"))
    xCoordSN[0] = xDelta/2
    for i in range(1, len(xCoordSN)):
        xCoordSN[i] = xCoordSN[i-1] + xDelta
    xSN, ySN = np.meshgrid(xCoordSN,yCoordSN)

    #Para las coordenadas de las caras oeste y este
    xCoordWE = np.linspace(0.0, xDimen, xPoints+1, endpoint = True, dtype = np.dtype("float64"))
    yCoordWE[0] = yDelta/2
    for i in range(1, len(yCoordWE)):
        yCoordWE[i] = yCoordWE[i-1] + yDelta

    xWE, yWE = np.meshgrid(xCoordWE,yCoordWE)
    fig = plt.figure()                                         
    ax = fig.add_subplot(111) 
    ax.plot(xWE, yWE, marker='o', color='k', linestyle='none')
    ax.plot(xSN, ySN, marker='o', color='b', linestyle='none')
    ax.set_xlim([0, xDimen])
    ax.set_ylim([0, yDimen])
    ax.set_title('Grid of the domain')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    #plt.show()
    return xDelta, yDelta, xSN, ySN, xWE, yWE, fig, ax 

def positionFaceCenter(ax, xPoints, yPoints, xDimen, yDimen, xDelta, yDelta):
    xCoord = np.linspace(xDelta/2, xDimen-xDelta/2, xPoints, endpoint = True, dtype = np.dtype("float64"))
    yCoord = np.linspace(yDelta/2, yDimen-yDelta/2, yPoints, endpoint = True, dtype = np.dtype("float64"))

    xCenter, yCenter = np.meshgrid(xCoord, yCoord)
    ax.plot(xCenter, yCenter, marker='o', color='g', linestyle='none')
    plt.show()
    return xCenter, yCenter

u = lambda theta, H, L, U0, x, y : U0*(y*np.exp(-(5*y/H)**2)*np.cos(3*np.pi*x/L) +\
                                       np.tanh(5/2*(2*y+H)/(theta*H)) - \
                                       np.tanh(5/2*(2*y-H)/(theta*H)) - 1)

v = lambda H, L, U0, x, y : -3*np.pi*U0*H**2/(50*L)*np.exp(-(5*y/H)**2)*np.sin(3*np.pi*x/L)

def G(H, c, lam, kappa, G0, x, y, t):
    Source = G0*np.exp(-(y-(H*np.sin(2*c*t)/3))**2/lam**2 - (x-x0*(c*t+1))**2/kappa**2 )
    return Source
#------------------------------------------------------------------------------------------

#Definimos la precisión para hacer los cálculos
precision = 'float64'
#Parámetros del perfil de velocidad
theta = np.float64(0.8)
U0 = np.float64(2.5)
#Parámetros para la concentración
G0 = np.float64(0.25) #[kg/m3]
x0 = np.float64(0.2) #[m]
y0 = np.float64(0.2) #[m]
kappa = np.float64(0.04) #[m]
lam = np.float64(0.04)
c = np.float64(0.3) #[s^-1]
D = np.float64(4e-6)
#Definimos el número de celdas en X y en Y
xPoints = np.int64(15)
yPoints = np.int64(10)
#Dimensiones del dominio
h = np.float64(2.0)
xDimen = np.float64(3*h)
yDimen = np.float64(h)
#Tiempo
simTime = 10
tDelta = 0.2
time = np.arange(0, simTime+tDelta, tDelta, dtype = np.dtype(precision))
#Adimensionalizo el tiempo
tCharact = h/U0
simTimeNon = np.float64(simTime/tCharact)
tDeltaNon = np.float64(tDelta/tCharact)

#vector para almacenar el término fuente
Source = np.zeros((yPoints, xPoints), dtype = np.dtype(precision))
phi = np.zeros((yPoints, xPoints), dtype = np.dtype(precision))

#Con esta función obtenngo las coordenadas x, y de los centros de caras. Está para 
# cara norte-sur y oeste-este
xDelta, yDelta, xSN, ySN, xWE, yWE, fig1, ax1 = facesCoordinates(xPoints, yPoints, xDimen, yDimen)
xCenter, yCenter = positionFaceCenter(ax1, xPoints, yPoints, xDimen, yDimen, xDelta, yDelta)

#Adimensionalizo el espacio. Las coordenadas ya están con formato meshgrid
xDimenNon = np.float64(xDimen/h)
yDimenNon = np.float64(yDimen/h)
xSNNon = np.float64(xSN/h)
ySNNon = np.float64(ySN/h)
xWENon = np.float64(xWE/h)
yWENon = np.float64(yWE/h)
xDeltaNon = np.float64(xDelta/h)
yDeltaNon = np.float64(yDelta/h)

xCenterNon = np.float64(xCenter/h)
yCenterNon = np.float64(yCenter/h)

#Grafico la malla con las dimensiones adimensionales para checkear
#fig2 = meshVisualization(xPoints, yPoints, xDimenNon, yDimenNon, xFaceCoordNon, yFaceCoordNon, xDeltaNon, yDeltaNon)

#Obtengo el perfil de velocidad. Le resto la mitad de la altura porque
#los perfiles están centrados en la mita de la altura.
vSN = v(yDimen, xDimen, U0, xSN, ySN-yDimen/2)
uWE = u(theta, yDimen, xDimen, U0, xWE, yWE-yDimen/2)
#Source = G(yDimenNon, c, lam, kappa, G0, xMeshGrid, yMeshGrid, t)

#Adimensionalizo los perfiles de velocidad
vSNNon = np.float64(vSN/U0)
uWENon = np.float64(uWE/U0)

#Constantes del problema
beta = np.float64(tDeltaNon/(2*xDeltaNon*yDeltaNon))
H = np.float64(h/(U0*G0))
Pe = np.float64(D/(h*U0))

#Matrices y vectores para soolucionar el problema
matAF = np.zeros((yPoints, xPoints), dtype = np.dtype("float64"))
vectBF = np.zeros((yPoints*xPoints, 1), dtype = np.dtype("float64"))
vectPhiF = np.zeros((yPoints*xPoints, 1), dtype = np.dtype("float64"))

matAP = np.zeros((yPoints, xPoints), dtype = np.dtype("float64"))
vectBP = np.zeros((yPoints*xPoints, 1), dtype = np.dtype("float64"))
vectPhiP = np.zeros((yPoints*xPoints, 1), dtype = np.dtype("float64"))

#Introduzco un conjunto de índices para la malla al estilo Hans Petter
Ix = range(xPoints)
Iy = range(yPoints)

a = lambda ue, vn, uw, vs, Pe, U0, yDelta, xDelta: -(ue*yDelta)/(2*U0) - (vn*xDelta)/(2*U0) + \
                                                    (uw*yDelta)/(2*U0) + (vs*xDelta)/(2*U0) - \
                                                    2*(Pe*yDelta)/(xDelta) -2*(Pe*xDelta)/(yDelta) 

b = lambda ue, xDelta, yDelta, Pe, U0: -(ue*yDelta)/(2*U0) + (Pe*yDelta)/(xDelta)

c = lambda vn, xDelta, yDelta, Pe, U0: -(vn*xDelta)/(2*U0) + (Pe*xDelta)/(yDelta)

d = lambda uw, xDelta, yDelta, Pe, U0: -(uw*yDelta)/(2*U0) + (Pe*yDelta)/(xDelta)

e = lambda vs, xDelta, yDelta, Pe, U0: -(vs*xDelta)/(2*U0) + (Pe*xDelta)/(yDelta)


def systemCN(matAF, vectBF, matAP, vectBP, Ix, Iy, beta, xDelta, yDelta, Pe, xPoints, yPoints, 
             U0, vSN, uWE, phiw, tDelta, H, c, lam, kappa, G0, xCenterNon, yCenterNon, time):
    
    Source = G(H, c, lam, kappa, G0, xCenterNon, yCenterNon, time)
    eqGlobal = 0
    for j in Iy:
        for i in Ix:
            constA = a(uWE[j, i+1], vSN[j+1, i], uWE[j, i], vSN[j, i], Pe, U0, yDelta, xDelta)
            constB = b(uWE[j, i+1], xDelta, yDelta, Pe, U0)
            constC = c(vSN[j+1, i], xDelta, yDelta, Pe, U0)
            constD = d(uWE[j, i], xDelta, yDelta, Pe, U0)
            constE = e(vSN[j, i], xDelta, yDelta, Pe, U0)
            #esquina inferior izquierda
            if  (j ==0 and i == 0):
                matAF[eqGlobal, eqGlobal] = np.float64(1 + beta*(-constA + constB - constE))
                matAP[eqGlobal, eqGlobal] = np.float64(1 + beta*(constA - constB + constE)) 

            #esquina inferior derecha
            if j == 0 and i == Ix[-1]:
                matAF[eqGlobal, eqGlobal] = np.float64(1 + beta*(-constA - constB - constE))
                matAP[eqGlobal, eqGlobal] = np.float64(1 + beta*(constA + constB + constE))

            #esquina superior derecha
            if j == Iy[-1] and i == Ix[-1]:
                matAF[eqGlobal, eqGlobal] = np.float64(1 + beta*(-constA - constB - constC))
                matAP[eqGlobal, eqGlobal] = np.float64(1 + beta*(constA + constB + constC))

            #esquina superior izquierda
            if j == Iy[-1] and i == 0:
                matAF[eqGlobal, eqGlobal] = np.float64(1 + beta*(-constA - constC - constD))
                matAP[eqGlobal, eqGlobal] = np.float64(1 + beta*(constA + constC + constD))

            #Gamma 4
            if i == 0:
                vectBF[eqGlobal] = np.float64(-2*beta*constD*phiw - (tDelta*H*Source[j,i])/2)
                vectBP[eqGlobal] = np.float64(2*beta*constD*phiw + (tDelta*H*Source[j,i])/2)

            #A la derecha
            if i > 0:
                matAF[eqGlobal, eqGlobal-1] = np.float64(-beta*constD)
                matAP[eqGlobal, eqGlobal-1] = np.float64(beta*constD)
            #A la izquierda
            if i < Ix[-1]:
                matAF[eqGlobal, eqGlobal+1] = np.float64(-beta*constB)
                matAP[eqGlobal, eqGlobal+1] = np.float64(beta*constB)
            #Por arriba de la inferior
            if j > 0:
                matAF[eqGlobal, eqGlobal-xPoints] = np.float64(-beta*constE)
                matAP[eqGlobal, eqGlobal-xPoints] = np.float64(beta*constE)
            #Por abajo de la superior
            if j < Iy[-1]:
                matAF[eqGlobal, eqGlobal+xPoints] = np.float64(-beta*constC)
                matAP[eqGlobal, eqGlobal+xPoints] = np.float64(beta*constC)




for t in range(1, len(time)):
    pass



