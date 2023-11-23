import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------
#Con esta función calculo coordenadas en una dimensión para los nodos
def positionsUniformGrid(points, dimension, coordinate, precision):
    if precision == "float32":
        delta = np.float32(dimension/(points-1)) 
    else:
        delta = np.float64(dimension/(points-1))
    for i in range(1, len(coordinate)):
        coordinate[i] = coordinate[i-1] + delta
    return delta, coordinate

#Visualizo la  malla. Sirve para parámetros dimensionales y no dimensionales
def meshVisualization(xPoints, yPoints, xDimen, yDimen, xCoordinate, yCoordinate):
    X, Y = np.meshgrid(xCoordinate, yCoordinate)
    fig = plt.figure()                                         
    ax = fig.add_subplot(111) 

    for i in range(0,xPoints-1):
        ax.plot(X[:,i], Y[:,0], color='k')

    for i in range(0,yPoints-1):
        ax.plot(X[0,:], Y[i,:], color='k')

    plt.xlim([0, xDimen])
    plt.ylim([0, yDimen])
    plt.title('Grid of the domain')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()
    return fig

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
lam = 0.04
c = 0.3 #[s^-1]
#Definimos el número de puntos en X y en Y
xPoints = np.int64(30)
yPoints = np.int64(15)
#Dimensiones del dominio
h = np.float64(2.0)
xDimen = np.float64(3*h)
yDimen = np.float64(h)
#Tiempo
simTime = 10
tDelta = 0.2
time = np.arange(0, simTime, tDelta, dtype = np.dtype(precision))
#Adimensionalizo el tiempo
tCharact = h/U0
simTimeNon = np.float64(simTime/tCharact)
tDeltaNon = np.float64(tDelta/tCharact)

#vectores para almacenar las coordenadass y visualizar la malla
xCoordinate = np.zeros((xPoints,),  dtype = np.dtype(precision))
yCoordinate = np.zeros((yPoints,),  dtype = np.dtype(precision))
#vectores para almacenar la velocidad
xVelocity = np.zeros((yPoints, xPoints), dtype = np.dtype(precision))
yVelocity = np.zeros((yPoints, xPoints), dtype = np.dtype(precision))
xVelocityNon = np.zeros((yPoints, xPoints), dtype = np.dtype(precision))
yVelocityNon = np.zeros((yPoints, xPoints), dtype = np.dtype(precision))
#vector para almacenar el término fuente
Source = np.zeros((yPoints, xPoints), dtype = np.dtype(precision))

xDelta, xCoordinate = positionsUniformGrid(xPoints, xDimen, xCoordinate, precision)
yDelta, yCoordinate = positionsUniformGrid(yPoints, yDimen, yCoordinate, precision)
#fig1 = meshVisualization(xPoints, yPoints, xDimen, yDimen, xCoordinate, yCoordinate)

#Adimensionalizo el espacio
xDimenNon = np.float64(xDimen/h)
yDimenNon = np.float64(yDimen/h)
xCoordinateNon = np.float64(xCoordinate/h)
yCoordinateNon = np.float64(yCoordinate/h)
xDeltaNon = np.float64(xDelta/h)
yDeltaNon = np.float64(yDelta/h)

#Grafico la malla con las dimensiones adimensionales para checkear
fig2 = meshVisualization(xPoints, yPoints, xDimenNon, yDimenNon, xCoordinateNon, yCoordinateNon)

#Utilizo el meshgrid para poder obtener los perfiles de velocidad y término fuente dimensionales
xMeshGrid, yMeshGrid = np.meshgrid(xCoordinate, np.flip(yCoordinate)-yDimen/2)
xVelocity = u(theta, yDimen, xDimen, U0, xMeshGrid, yMeshGrid)
yVelocuty = v(yDimenNon, xDimenNon, U0, xMeshGrid, yMeshGrid)
#Source = G(yDimenNon, c, lam, kappa, G0, xMeshGrid, yMeshGrid, t)

#Adimensionalizo los perfiles de velocidad
xVelocityNon = xVelocity/U0
yVelocityNon = yVelocity/U0


