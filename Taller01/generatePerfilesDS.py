#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import time

# START Definicion de funciones
def profileS(posX,xFrac,maxS,sigma):
    xMin = np.float64(posX[0])
    xMax = np.float64(posX[-1])
    totalL = xMax - xMin
    return maxS * np.exp(-0.5*(np.power((posX-xFrac*totalL)/sigma,2.0)))

def profileD(posX,minD):
    xMin = np.float64(posX[0])
    xMax = np.float64(posX[-1])
    totalL = xMax - xMin
    return minD + 0.05*(np.sin(2.5*posX*np.pi/totalL)+1.0)
# END Definicion de funciones

def main():
    # Lectura de datos entrada
    simData = pd.read_csv("simDataDifConvReacDual1D.dat")
    simData.to_numpy()
    simData.columns = simData.columns.str.strip()
    simData.set_index('Variable',inplace=True)
    domainL  = np.float64(simData.loc['domainL']['Value'])
    nNodesX  = np.int64(simData.loc['nNodesX']['Value'])
    meshExpRat  = np.float64(simData.loc['meshExpRat']['Value'])
    posSPeak  = np.float64(simData.loc['posSPeak']['Value'])
    maxS  = np.float64(simData.loc['maxS']['Value'])    
    sigmaS  = np.float64(simData.loc['sigmaS']['Value'])
    minDiffVal  = np.float64(simData.loc['minDiffVal']['Value'])

    positionsX = np.zeros(int(nNodesX),dtype=np.float64)
    deltasX=np.zeros(int(nNodesX),dtype=np.float64)
    # Valores posicionales se calculan internamente
    if (np.abs(meshExpRat-1.0E0)<1.0E-10):
        startDeltaX = domainL/np.float64(nNodesX-1)
        deltasX[:] = startDeltaX
    else:
        startDeltaX = domainL*(meshExpRat - 1.0)/(np.power(meshExpRat,int(nNodesX)-1)-1.0)
        deltasX[0] = startDeltaX/meshExpRat
        for i in range(1,int(nNodesX)):
          deltasX[i] = np.power(meshExpRat,i-1)*startDeltaX
    
    for i in range(1,int(nNodesX)):
      positionsX[i] = positionsX[i-1] + deltasX[i]
    
    # Generamos perfiles y los guardamos en archivo,
    # luego los volvemos a adquirir con lectura de CSV en el programa principal.
    valsS = profileS(positionsX,posSPeak,maxS,sigmaS)
    fig = plt.figure()
    plt.plot(positionsX,valsS,'o-')
    plt.grid()
    plt.show()
    valsD = profileD(positionsX,minDiffVal)
    fig = plt.figure()
    plt.plot(positionsX,valsD,'o-')
    plt.grid()
    plt.show()

    # Escribiendo valores en archivo perfilesDS.dat
    perfilesDS = np.zeros((int(nNodesX),2))
    perfilesDS[:,0] = np.transpose(valsD)
    perfilesDS[:,1] = np.transpose(valsS)
    perfilesDSAsDF=pd.DataFrame(perfilesDS)
    perfilesDSAsDF.to_csv('./perfilesDS.dat',index=False,header=False,index_label=False)

if __name__ == "__main__":
    main()


