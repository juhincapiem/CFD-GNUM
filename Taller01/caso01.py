import numpy as np
import matplotlib.pyplot as plt

fun = lambda k0, k1, k2, x :(1.1*np.cos(k1*x) - 1.3*np.sin(k2*x))*np.exp(-k0*x**2)

Der1F = lambda k0, k1, k2, x : np.exp(-k0*x**2)*(26*k0*x*np.sin(k2*x) - \
                               13*k2*np.cos(k2*x) - 11*k1*np.sin(k1*x) - 22*k0*x*np.cos(k1*x))/10

Der2F = lambda k0, k1, k2, x : -np.exp(-k0*x**2)*((52*k0**2*x**2 - 13*k2**2 - 26*k0)*np.sin(k2*x) - 52*k0*k2*x*np.cos(k2*x) \
                                - 44*k0*k1*x*np.sin(k1*x) + (-44*k0**2*x**2 + 11*k1**2 + 22*k0)*np.cos(k1*x))/10

def firtOrder(nodes, xDelta, precision):
        A1 = np.eye(nodes,nodes, k=0, dtype = np.dtype(precision))*-1 +\
            np.eye(nodes,nodes, k=1, dtype = np.dtype(precision))*1 
        A1[-1,-1] = 1; A1[-1,-2] = -1; A1 = A1*(1/xDelta); A1 = A1.astype(dtype = np.dtype(precision))
        return A1

def secondOrder(nodes, xDelta, precision):
        A2 = np.eye(nodes,nodes, k=1, dtype = np.dtype(precision))*1 +\
            np.eye(nodes,nodes, k=-1, dtype = np.dtype(precision))*-1  
        border = np.array([-3, 4, -1], dtype=np.dtype(precision))
        A2[0,:len(border)] = border
        A2[-1,-len(border):] = np.flip(border)*-1
        A2 = A2*(1/(2*xDelta)); A2 = A2.astype(dtype = np.dtype(precision))
        return A2

def fourthOrder(nodes, xDelta, precision):
        A4 = np.eye(nodes,nodes, k=0, dtype = np.dtype(precision))*4 +\
             np.eye(nodes,nodes, k= 1, dtype = np.dtype(precision))*1 +\
             np.eye(nodes,nodes, k=-1, dtype = np.dtype(precision))*1  
        A4[0,1], A4[-1,-2], A4[0,0], A4[-1,-1] = 0, 0, 1, 1 
        A4 = A4.astype(dtype = np.dtype(precision))
        A4_inv = np.linalg.inv(A4)

        B4 = np.eye(nodes, nodes, k=1, dtype = np.dtype(precision))*3 +\
             np.eye(nodes, nodes, k=-1, dtype = np.dtype(precision))*-3
        border = np.array([-25/12, 48/12,-36/12, 16/12, -3/12], dtype=np.dtype(precision))
        B4[0,:len(border)] = border
        B4[-1, -len(border):] = np.flip(border)*-1
        B4 = B4*(1/(xDelta))
        
        return A4_inv, B4

def numDer_orderN(nodes, precision, A, F, B=1, typeScheme="normal"):
        if typeScheme == "normal":
                derN_orderN = np.zeros(nodes, dtype = np.dtype(precision)); derN_orderN = np.reshape(derN_orderN, (nodes,1))
                derN_orderN = np.dot(A,F)
        elif typeScheme == "compact":
                derN_orderN = np.zeros(nodes, dtype = np.dtype(precision)); derN_orderN = np.reshape(derN_orderN, (nodes,1))
                derN_orderN = np.dot(np.dot(A,B),F)

        return derN_orderN

def firstDerivative(precision, numbDeltas, k0, k1, k2, type = "firstDer"):

    xDelta = np.zeros((1,numbDeltas),  dtype = np.dtype(precision))
    xDelta = np. reshape(xDelta,(1, numbDeltas))

    if precision == "float16":
        for i in range(1,numbDeltas+1):
            xDelta[0,i-1] = np.float16(np.pi/2**(i+2))
    elif precision == "float32":
        for i in range(1,numbDeltas+1):
            xDelta[0,i-1] = np.float32(np.pi/2**(i+2))
    else:
        for i in range(1,numbDeltas+1):
            xDelta[0,i-1] = np.float64(np.pi/2**(i+2))
            
    Data = np.zeros((len(xDelta[0]), 3), dtype = np.dtype(precision))
    Data = np.reshape(Data, (len(xDelta[0]), 3))

    for j in range(len(xDelta[0])):
        if precision == "float16":
            nodes = np.int_(np.ceil(np.float16(np.pi*2)/xDelta[0,j])) + 1
        elif precision == "float32":
            nodes = np.int_(np.ceil(np.float32(np.pi*2)/xDelta[0,j])) + 1
        else:
            nodes = np.int_(np.ceil(np.float64(np.pi*2)/xDelta[0,j])) + 1

        xCoord = np.linspace(-np.pi, np.pi, nodes, dtype = np.dtype(precision))
        
        #Create the F column vector that contain the values of the function
        F = np.zeros(nodes, dtype = np.dtype(precision))
        F = fun(k0, k1, k2, xCoord); F = np.reshape(F, (nodes,1))

        if type == "firstDer":
            #First order
            A1 = firtOrder(nodes, xDelta[0,j], precision)
            #Second order
            A2 = secondOrder(nodes, xDelta[0,j], precision)
            #Fourth order
            A4_inv, B4 = fourthOrder(nodes, xDelta[0,j], precision)
            #Compute the derivative
            der1_order1 = numDer_orderN(nodes, precision, A1, F)
            der1_order2 = numDer_orderN(nodes, precision, A2, F)
            der1_order4 = numDer_orderN(nodes, precision, A4_inv, F, B4, typeScheme="compact")
            #Compute L2 error
            der1Analy = np.reshape(Der1F(k0, k1, k2, xCoord), (nodes,1))
            Data[j,0] = np.linalg.norm(der1Analy-der1_order1)
            Data[j,1] = np.linalg.norm(der1Analy-der1_order2)
            Data[j,2] = np.linalg.norm(der1Analy-der1_order4)
        elif type == "secondDer":
            #First order
            A1 = firtOrder(nodes, xDelta[0,j], precision)
            #Second order
            A2 = secondOrder(nodes, xDelta[0,j], precision)
            #Compute the derivative
            der2_order1 = numDer_orderN(nodes, precision, np.dot(A1,A1), F)
            der2_order2 = numDer_orderN(nodes, precision, np.dot(A2,A2), F)
            #Compute L2 error
            der2Analy = np.reshape(Der2F(k0, k1, k2, xCoord), (nodes,1))
            Data[j,0] = np.linalg.norm(der2Analy-der2_order1)
            Data[j,1] = np.linalg.norm(der2Analy-der2_order2)
            
    return xDelta, Data

def plotter(xDelta1, Data1, xDelta2, Data2, numDeltas, precision, type = "firstDer"):
    Dict = {}
    Dict[0] = {'Data': Data1, 'xDelta': xDelta1}
    Dict[1] = {'Data': Data2, 'xDelta': xDelta2}
    letter = ["a", "b"]
    i = 0
    fig, ax = plt.subplots(1,2, figsize=(10, 5), sharex = True, sharey = True)
    for info in Dict:
        xStep = np.reshape(Dict[info]["xDelta"], (numDeltas,1))
        
        ax[i].set_yscale("log")
        ax[i].set_xscale("log")
        ax[i].plot(xStep, np.reshape(Dict[info]["Data"][:,0], (numDeltas, 1)), "--bo", label='$1^{er}$ orden')
        ax[i].plot(xStep, np.reshape(Dict[info]["Data"][:,1], (numDeltas, 1)), "--go", label='$2^{do}$ orden')
        if type == "firstDer":
            ax[i].plot(xStep, np.reshape(Dict[info]["Data"][:,2], (numDeltas, 1)), "--ko", label='$4^{to}$ orden')
            ax[i].set_title("$1^{er}$, $2^{do}$ y $4^{to}$ orden, precisión "+ precision[i])
            fig.suptitle('Análisis de errores y precisión de esquemas FDM  para primera derivada')
            ax[i].set_xlabel("Tamaño de la malla $\Delta x$ \nFigura 1" + letter[i] +\
                             ": $1^{er}$, $2^{do}$ y $4^{to}$ orden, precisión"  + precision[i])
        elif type == "secondDer":
            ax[i].set_title("$1^{er}$ y $2^{do}$ orden, precisión "+ precision[i])
            fig.suptitle('Análisis de errores y precisión de esquemas FDM  para segunda derivada')
            ax[i].set_xlabel("Tamaño de la malla $\Delta x$ \nFigura 2" + letter[i] +\
                             ": $1^{er}$ y $2^{do}$ orden, precisión "  + precision[i])

        
        ax[i].set_ylabel("Error $L_2$ ")
        ax[i].legend()
        ax[i].grid()
        i+=1
    return ax
    
k0 = 2/3; k1 = 5/3; k2 = 3*k1
numDeltas = 10
precision = np.array(['float32', 'float64'])
xDelta1, Data1 = firstDerivative(precision[0], numDeltas, k0, k1, k2)
xDelta2, Data2 = firstDerivative(precision[1], numDeltas, k0, k1, k2)


ax = plotter(xDelta1, Data1, xDelta2, Data2, numDeltas, precision, type = "firstDer")

numDeltas = 8
precision = np.array(['float16', 'float64'])
xDelta3, Data3 = firstDerivative(precision[0], numDeltas, k0, k1, k2, type = "secondDer")
#xDelta3, Data3 = firstDerivative(precision[1], numDeltas, k0, k1, k2, type = "secondDer")
xDelta4, Data4 = firstDerivative(precision[1], numDeltas, k0, k1, k2, type = "secondDer")

ax = plotter(xDelta3, Data3, xDelta4, Data4, numDeltas, precision, type = "secondDer")