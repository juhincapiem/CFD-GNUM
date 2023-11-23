import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from icecream import ic
import time

# Lectura de datos entrada
simData = pd.read_csv("simDataDifConvReacDual1D.dat")
simData.to_numpy()
simData.columns = simData.columns.str.strip()
simData.set_index('Variable',inplace=True)
print(simData)
domainL  = np.float64(simData.loc['domainL']['Value'])
flowVelU  = np.float64(simData.loc['flowVelU']['Value'])
decayTau  = np.float64(simData.loc['decayTau']['Value'])
valPQAtL  = np.float64(simData.loc['valPQAtL']['Value'])
nNodesX  = np.float64(simData.loc['nNodesX']['Value'])
meshExpRat  = np.float64(simData.loc['meshExpRat']['Value'])
realSimTime  = np.float64(simData.loc['realSimTime']['Value'])
realDeltaT  = np.float64(simData.loc['realDeltaT']['Value'])
nPlots = np.float64(simData.loc['nPlots']['Value'])

#Definir los valores kappa
kappa1 = np.float64(1.0/(flowVelU*domainL))
kappa2 = np.float64(decayTau*domainL/flowVelU)
kappa3 = np.float64(domainL/(flowVelU*valPQAtL))

# ----------------------------------------------------------------
#Adimensionalización del tiempo: tiempo característico, deltaT adimensional
#y tiempo total de simulación adimensional
caracteristicTime = domainL/flowVelU
deltaTnonDim = np.float64(realDeltaT/caracteristicTime)
nonDimSimTime = np.float64(realSimTime/caracteristicTime)

# ----------------------------------------------------------------
# Definicion numero de pasos de tiempo usando tiempo total de simulación adimensional
#y el paso de tiempo adimensional. Se suma uno para no icnluir las condiciones iniciales
#como parte de la simulación total
nTstep  = np.int64((nonDimSimTime/deltaTnonDim)+1.0E0)
nTotalPoints = np.int64(nNodesX)
#No se usan todos los nodos ya que siempre conocemos el valor del último nodo por las
#condición de Dirichlet
nSolucionPoints = np.int64(nNodesX-1.0)

# ----------------------------------------------------------------
positionsX = np.zeros(nTotalPoints, dtype= np.float64)
deltasX = np.zeros(nTotalPoints, dtype= np.float64)

#Para malla uniforme
if (np.abs(meshExpRat-1.0E0)<1.0E-10):
    startDeltaX = domainL/np.float64(nNodesX-1)
    deltasX[:] = startDeltaX
#Para malla NO uniforme
else:
    startDeltaX = domainL*(meshExpRat - 1.0)/(np.power(meshExpRat,int(nNodesX)-1)-1.0)
    deltasX[0] = startDeltaX/meshExpRat
    for i in range(1,int(nNodesX)):
        deltasX[i] = np.power(meshExpRat,i-1)*startDeltaX

#La suma empieza desde la posición 1 porque la posición 0 es un delta X 
#ficiticio que se usa más adelante
print("La longitud total obtenida de la suma de intervalos es: ",deltasX[1:].sum())

#Sin importar si es malla uniforme o no, este ciclo siempre es vigente
for i in range(1,nTotalPoints):
  positionsX[i] = positionsX[i-1] + deltasX[i]

# ----------------------------------------------------------------
#Ahora se convierten los valores de posición en no dimensionales
startDeltaXNonDim = startDeltaX/domainL
deltasXNonDim = deltasX/domainL
positionsXNonDim = positionsX/domainL
print("La longitud total adimensional obtenida de la suma de intervalos es: ",deltasXNonDim[1:].sum())

# ----------------------------------------------------------------
# Adquisicion de valores de coeficiente de Difusion (D) y de la función Fuente (S)
dataFile = "perfilesDS.dat"
dataRawDS = pd.read_csv(dataFile,delim_whitespace=False,
                        skip_blank_lines=True,header=None,dtype=np.float64)

dataFiltered = dataRawDS.iloc[:,[0,1]]
dataFiltered.to_numpy()
dataD = np.float64(dataFiltered.iloc[:,0])
dataS = np.float64(dataFiltered.iloc[:,1])
kappa4 = flowVelU*domainL/dataD[0]

# ----------------------------------------------------------------
#Se requiere Nodos-1 espacios en los vectores, dado que no se va a resolver para el 
# último nodo
valsAlpha = np.zeros(nSolucionPoints,dtype=np.float64)
valsBeta  = np.zeros(nSolucionPoints,dtype=np.float64)
valsGamma = np.zeros(nSolucionPoints,dtype=np.float64)
valsEta = np.zeros(nSolucionPoints,dtype=np.float64)

#Aquí se usa el delta X ficticio
valsAlpha[0] = deltaTnonDim * kappa1 * 0.5*(dataD[1]+dataD[0]) / \
        ((deltasXNonDim[1]+deltasXNonDim[0])*deltasXNonDim[1])
valsBeta[0] = deltaTnonDim * kappa1 * 0.5*(dataD[0]+dataD[0]) / \
        ((deltasXNonDim[1]+deltasXNonDim[0])*deltasXNonDim[0])
valsGamma[0] = deltaTnonDim/(2.0*(deltasXNonDim[1]+deltasXNonDim[0]))

for i in range(1,int(nNodesX-1)):
    valsAlpha[i] = deltaTnonDim * kappa1 * 0.5*(dataD[i+1]+dataD[i]) / \
            ((deltasXNonDim[i+1]+deltasXNonDim[i])*deltasXNonDim[i+1])
    valsBeta[i] = deltaTnonDim * kappa1 * 0.5*(dataD[i]+dataD[i-1]) / \
            ((deltasXNonDim[i+1]+deltasXNonDim[i])*deltasXNonDim[i])
    valsGamma[i] = deltaTnonDim/(2.0*(deltasXNonDim[i+1]+deltasXNonDim[i]))

valDelta = 0.5E0*kappa2*deltaTnonDim
valEta   = 0.5E0*kappa3*deltaTnonDim

coefsA = np.zeros(nSolucionPoints,dtype=np.float64)
coefsB1P = np.zeros(nSolucionPoints,dtype=np.float64)
coefsB2P = np.zeros(nSolucionPoints,dtype=np.float64)
coefsC = np.zeros(nSolucionPoints,dtype=np.float64)
coefsB1Q = np.zeros(nSolucionPoints,dtype=np.float64)
coefsB2Q = np.zeros(nSolucionPoints,dtype=np.float64)

for i in range(nSolucionPoints):
    coefsA[i] = valsBeta[i] + valsGamma[i]
    coefsB1P[i] = 1.0E0 + valsAlpha[i] + valsBeta[i] + valDelta
    coefsB1Q[i] = 1.0E0 + valsAlpha[i] + valsBeta[i] 
    coefsB2P[i] = 1.0E0 - valsAlpha[i] - valsBeta[i] - valDelta 
    coefsB2Q[i] = 1.0E0 - valsAlpha[i] - valsBeta[i] 
    coefsC[i] = valsAlpha[i] - valsGamma[i]


# Matriz de Coeficientes de Vector valores futuros. 
# Dado que BC en x=L es Dirichlet, solo se necesitan nX-1 valores
matCoefAP = np.zeros((nSolucionPoints,nSolucionPoints),dtype=np.float64)
matCoefAQ = np.zeros((nSolucionPoints,nSolucionPoints),dtype=np.float64)
# Matriz de Coeficientes de Vector valores actuales. 
# Dado que BC en x=L es Dirichlet, solo se necesitan nX-1 valores
matCoefBP = np.zeros((nSolucionPoints,nSolucionPoints),dtype=np.float64)
matCoefBQ = np.zeros((nSolucionPoints,nSolucionPoints),dtype=np.float64)

#Completamos la primera línea para la matriz A y B tanto para P como para Q
matCoefAP[0,0] = coefsB1P[0] + coefsA[0]*kappa4*(deltasXNonDim[1] + deltasXNonDim[0])
matCoefAP[0,1] = -1.0E0*(coefsA[0]+coefsC[0])
matCoefBP[0,0] = coefsB2P[0] - coefsA[0]*kappa4*(deltasXNonDim[1] + deltasXNonDim[0])
matCoefBP[0,1] = +1.0E0*(coefsA[0]+coefsC[0])

matCoefAQ[0,0] = coefsB1Q[0] + coefsA[0]*kappa4*(deltasXNonDim[1] + deltasXNonDim[0])
matCoefAQ[0,1] = -1.0E0*(coefsA[0]+coefsC[0])
matCoefBQ[0,0] = coefsB2Q[0] - coefsA[0]*kappa4*(deltasXNonDim[1] + deltasXNonDim[0])
matCoefBQ[0,1] = +1.0E0*(coefsA[0]+coefsC[0])

#Llenamos el centro de la matriz A y B
for i in range(1,nSolucionPoints-1):
    matCoefAP[i,i-1] = -1.0E0*coefsA[i]
    matCoefAP[i,i  ] =  1.0E0*coefsB1P[i]
    matCoefAP[i,i+1] = -1.0E0*coefsC[i]
    matCoefBP[i,i-1] =  1.0E0*coefsA[i]
    matCoefBP[i,i  ] =  1.0E0*coefsB2P[i]
    matCoefBP[i,i+1] =  1.0E0*coefsC[i]
    # ------ Ahora matrices para Q
    matCoefAQ[i,i-1] = -1.0E0*coefsA[i]
    matCoefAQ[i,i  ] =  1.0E0*coefsB1Q[i]
    matCoefAQ[i,i+1] = -1.0E0*coefsC[i]
    matCoefBQ[i,i-1] =  1.0E0*coefsA[i]
    matCoefBQ[i,i  ] =  1.0E0*coefsB2Q[i]
    matCoefBQ[i,i+1] =  1.0E0*coefsC[i]

matCoefAP[-1,-2] = -1.0E0*coefsA[nSolucionPoints-1]
matCoefAP[-1,-1] =  1.0E0*coefsB1P[nSolucionPoints-1]
matCoefBP[-1,-2] =  1.0E0*coefsA[nSolucionPoints-1]
matCoefBP[-1,-1] =  1.0E0*coefsB2P[nSolucionPoints-1]

matCoefAQ[-1,-2] = -1.0E0*coefsA[nSolucionPoints-1]
matCoefAQ[-1,-1] =  1.0E0*coefsB1Q[nSolucionPoints-1]
matCoefBQ[-1,-2] =  1.0E0*coefsA[nSolucionPoints-1]
matCoefBQ[-1,-1] =  1.0E0*coefsB2Q[nSolucionPoints-1]

# Vector de terminos Fuente
vecSources = np.zeros(nTotalPoints,dtype=np.float64)
vecSources[:] = 2.0E0*valEta*dataS[:]

# Creacion de matriz para almacenamiento de la solucion en el tiempo
solucionP = np.zeros((nTotalPoints,nTstep),dtype=np.float64)
solucionQ = np.zeros((nTotalPoints,nTstep),dtype=np.float64)

vecRHSP = np.zeros(nSolucionPoints,dtype=np.float64)
vecSolP = np.zeros(nSolucionPoints,dtype=np.float64)
vecBCsP = np.zeros(nSolucionPoints,dtype=np.float64)

vecRHSQ = np.zeros(nSolucionPoints,dtype=np.float64)
vecSolQ = np.zeros(nSolucionPoints,dtype=np.float64)
vecBCsQ = np.zeros(nSolucionPoints,dtype=np.float64)

vecBCsP[:] = vecSources[:-1]
# El 1.0 al final de la expresión es para recordar que este es el valor de P en x=L
vecBCsP[-1] = vecBCsP[-1] + 2.0E0*coefsC[-1]*1.0E0 
solucionP[-1,0]=1.0E0
# Para Q es ligeramente diferente, verificar con formulacion:
vecBCsQ[-1] = 2.0E0*coefsC[-1]*1.0E0
solucionQ[-1,0] = 1.0E0
# ----------------------------------------------------------------
startTime = time.time()
# ----------------------------------------------------------------
invMatCoefAP = np.linalg.inv(matCoefAP)
invMatCoefAQ = np.linalg.inv(matCoefAQ)
# ----------------------------------------------------------------
vecSolP[:] = solucionP[:-1,0]
vecSolQ[:] = solucionQ[:-1,0]

# Este caso si requiere avance temporal
for t in range(1,nTstep):
    # Solucion para P(x,t).
    # Producto Matriz-Vector lado derecho y suma cond. frontera
    vecRHSP = np.dot(matCoefBP,vecSolP)+vecBCsP
    vecBCsQ = valDelta*vecSolP
    # Producto inversa matriz izquierda por vector lado derecho
    vecSolP = np.dot(invMatCoefAP,vecRHSP)

    # Ahora ajuste de vector condiciones de frontera para Q
    vecBCsQ+= valDelta*vecSolP
    vecBCsQ[-1] = vecBCsQ[-1] + 2.0E0*coefsC[-1]*1.0E0
    # Solucion para Q(x,t).
    # Producto Matriz-Vector lado derecho y suma cond. frontera
    vecRHSQ = np.dot(matCoefBQ,vecSolQ)+vecBCsQ
    # Producto inversa matriz izquierda por vector lado derecho
    vecSolQ = np.dot(invMatCoefAQ,vecRHSQ)

    # Almacenamiento de valores de P(x) por paso de tiempo << Mucha memoria -- solo ilustrativo >>
    solucionP[:-1,t] = vecSolP
    solucionQ[:-1,t] = vecSolQ
    # Introduction de Condiciones de Frontera
    solucionP[-1,t] =1.0E0
    solucionQ[-1,t] =1.0E0

endTime = time.time()
print("Execution time of the program is: ", endTime-startTime)

# Defining colors and linestyles
default_cycler = (	cycler(linestyle=['-']) *
				  	cycler(marker=['o','s','^','d']) *
					cycler(color=['k', 'g', 'b', 'y','c','m']) 
				 )

plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=default_cycler)
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# Constructing plots
nMarkers = 30
if (nTotalPoints > 100):
    nEveryStride = ic(np.int64(nTotalPoints/nMarkers))
else:
    nEveryStride = ic(np.int64(nTotalPoints/2))

lines = []
nPlots = np.int64(nPlots)
deltaTPlot = np.float64(nonDimSimTime/np.float64(nPlots))
print('deltaTPlot=',deltaTPlot)
fig,ax = plt.subplots(1, 2, figsize=(12,7))
#lineStyle = ['k-o',]
for t in range(nPlots):
    timePlot = np.float64( t * deltaTPlot)
    posT = np.int64(timePlot/deltaTnonDim)
    labelText = f'$P(x)$, t = {timePlot:.3f}'
    if t == nPlots-1:
        if (posT != (nTstep-1)):
            posT = np.int64(-1)
            labelText = f'$P(x)$, t = {nonDimSimTime:.3f}'
            line = ax[0].plot(positionsXNonDim,solucionP[:,posT],
                        label=labelText,
                        markevery=nEveryStride,
                        markersize=5)
            lines.append(line)
    else:
        line = ax[0].plot(positionsXNonDim,solucionP[:,posT],
                    label=labelText,
                    markevery=nEveryStride,
                    markersize=5)
    lines.append(line)

# ----------------------------------------------------------------

# Reportando ahora datos en forma dimensional
# Constructing plots
lines = []
nPlots = np.int64(nPlots);
deltaTPlot = np.float64(realSimTime/np.float64(nPlots))
#fig2,ax2 = plt.subplots(figsize=(13,7))
lineStyle = ['k-o',]
print('Producing plot with dimensions')
for t in range(nPlots):
    timePlot = np.float64( t * deltaTPlot)
    posT = np.int64(timePlot/realDeltaT)
    labelText = f'$P(x)$, t = {timePlot:.3f}'
    if t == nPlots-1:
        if (posT != (nTstep-1)):
            posT = np.int64(-1)
            labelText = f'$P(x)$, t = {realSimTime:.3f}'
            line = ax[1].plot(positionsX,valPQAtL*solucionP[:,posT],
                        label=labelText,
                        markevery=nEveryStride,
                        markersize=5)
            lines.append(line)
    else:
        line = ax[1].plot(positionsX,valPQAtL*solucionP[:,posT],
                    label=labelText,
                    markevery=nEveryStride,
                    markersize=5)
    lines.append(line)


fig.suptitle('Cantidad de reactivo P a lo largo del río en diferentes tiempos')
ax[1].set_title('Cantidad de reactivo P dimensional')
ax[1].set_ylabel('$P~[kg]$')
ax[1].set_xlabel('$x~[m]$')

ax[0].set_title('Cantidad de reactivo P adimensional')
ax[0].set_ylabel('$P^*$')
ax[0].set_xlabel('$x^*$')

ax[1].legend()
ax[1].grid()
ax[0].legend()
ax[0].grid()
plt.show()
# ----------------------------------------------------------------
# ----------------------------------------------------------------

lines = []
nPlots = np.int64(nPlots)
deltaTPlot = np.float64(nonDimSimTime/np.float64(nPlots))
print('deltaTPlot=',deltaTPlot)
fig2,ax2 = plt.subplots(1, 2, figsize=(12,7))
#lineStyle = ['k-o',]
for t in range(nPlots):
    timePlot = np.float64( t * deltaTPlot)
    posT = np.int64(timePlot/deltaTnonDim)
    labelText = f'$P(x)$, t = {timePlot:.3f}'
    if t == nPlots-1:
        if (posT != (nTstep-1)):
            posT = np.int64(-1)
            labelText = f'$P(x)$, t = {nonDimSimTime:.3f}'
            line = ax2[0].plot(positionsXNonDim,solucionQ[:,posT],
                        label=labelText,
                        markevery=nEveryStride,
                        markersize=5)
            lines.append(line)
    else:
        line = ax2[0].plot(positionsXNonDim,solucionQ[:,posT],
                    label=labelText,
                    markevery=nEveryStride,
                    markersize=5)
    lines.append(line)

# ----------------------------------------------------------------

# Reportando ahora datos en forma dimensional
# Constructing plots
lines = []
nPlots = np.int64(nPlots);
deltaTPlot = np.float64(realSimTime/np.float64(nPlots))
#fig2,ax2 = plt.subplots(figsize=(13,7))
lineStyle = ['k-o',]
print('Producing plot with dimensions')
for t in range(nPlots):
    timePlot = np.float64( t * deltaTPlot)
    posT = np.int64(timePlot/realDeltaT)
    labelText = f'$P(x)$, t = {timePlot:.3f}'
    if t == nPlots-1:
        if (posT != (nTstep-1)):
            posT = np.int64(-1)
            labelText = f'$P(x)$, t = {realSimTime:.3f}'
            line = ax2[1].plot(positionsX,valPQAtL*solucionQ[:,posT],
                        label=labelText,
                        markevery=nEveryStride,
                        markersize=5)
            lines.append(line)
    else:
        line = ax2[1].plot(positionsX,valPQAtL*solucionQ[:,posT],
                    label=labelText,
                    markevery=nEveryStride,
                    markersize=5)
    lines.append(line)

fig2.suptitle('Cantidad de reactivo Q a lo largo del río en diferentes tiempos')
ax2[1].set_title('Cantidad de reactivo Q dimensional')
ax2[1].set_ylabel('$Q~[kg]$')
ax2[1].set_xlabel('$x~[m]$')

ax2[0].set_title('Cantidad de reactivo Q adimensional')
ax2[0].set_ylabel('$Q^*$')
ax2[0].set_xlabel('$x^*$')

ax2[1].legend()
ax2[1].grid()
ax2[0].legend()
ax2[0].grid()
plt.show()

perfilesPQ = np.zeros((int(nNodesX),3))
perfilesPQ[:,0] = np.transpose(positionsX)
perfilesPQ[:,1] = np.transpose(valPQAtL*solucionP[:,-1])
perfilesPQ[:,2] = np.transpose(valPQAtL*solucionQ[:,-1])
perfilesPQAsDF=pd.DataFrame(perfilesPQ)
perfilesPQAsDF.to_csv('./perfilesPQ.dat',index=False,header=False,index_label=False)