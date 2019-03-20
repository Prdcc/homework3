"""M345SC Homework 3, part 1
Enrico Ancilotto
01210716
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import hann, welch
import scipy.sparse as sp
from scipy.linalg import solve_banded
import os
import time
import seaborn as sns
import pandas as pd

def loadG(beta=True):
    return np.load("gB.npy") if beta else np.load("gA.npy")

def nwave(alpha,beta,Nx=256,Nt=801,T=200,display=True):
    """
    Question 1.1
    Simulate nonlinear wave model

    Input:
    alpha, beta: complex model parameters
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of |g| when true

    Output:
    g: Complex Nt x Nx array containing solution
    """

    #generate grid
    L = 100
    n = np.arange(-Nx/2,Nx/2)
    n = np.fft.fftshift(n)
    k = -(2*np.pi*n/L)**2
    
    def RHS(f,t,alpha,beta):
        """Computes dg/dt for model eqn.,
        f[:N] = Real(g), f[N:] = Imag(g)
        Called by odeint below
        """
        g = f[:Nx]+1j*f[Nx:]
        c = np.fft.fft(g)
        d2g = np.fft.ifft(k*c)

        dgdt = alpha*d2g + g -beta*g*g*g.conj()
        df = np.zeros(2*Nx)
        df[:Nx] = dgdt.real
        df[Nx:] = dgdt.imag
        return df

    #set initial condition
    g0 = np.random.rand(Nx)*0.1*hann(Nx)
    f0=np.zeros(2*Nx)
    f0[:Nx]=g0
    t = np.linspace(0,T,Nt)

    #compute solution
    f = odeint(RHS,f0,t,args=(alpha,beta))
    g = f[:,:Nx] + 1j*f[:,Nx:]

    if display:
        x = np.linspace(0,L,Nx+1)
        x = x[:-1]
        plt.figure()
        plt.contour(x,t,g.real)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of Real(g)')
        plt.show()

    return g

def nwaveNoise(alpha,beta,Nx=256,Nt=801,T=200,display=True):
    """
    Question 1.1
    Simulate nonlinear wave model

    Input:
    alpha, beta: complex model parameters
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of |g| when true

    Output:
    g: Complex Nt x Nx array containing solution
    """

    #generate grid
    L = 100
    n = np.arange(-Nx/2,Nx/2)
    n = np.fft.fftshift(n)
    k = -(2*np.pi*n/L)**2
    
    def RHS(f,t,alpha,beta):
        """Computes dg/dt for model eqn.,
        f[:N] = Real(g), f[N:] = Imag(g)
        Called by odeint below
        """
        g = f[:Nx]+1j*f[Nx:]
        c = np.fft.fft(g)
        d2g = np.fft.ifft(k*c)

        dgdt = alpha*d2g + g -beta*g*g*g.conj()
        df = np.zeros(2*Nx)
        df[:Nx] = dgdt.real
        df[Nx:] = dgdt.imag
        print(t)
        return df

    #set initial condition
    randomVec = np.random.rand(Nx)
    randomVecNoise = randomVec + np.random.normal(0,1e-3,(Nx,))

    g0 = randomVec*0.1*hann(Nx)
    g0Noise = randomVecNoise*0.1*hann(Nx)

    f0=np.zeros(2*Nx)
    f0[:Nx]=g0
    f0Noise=np.zeros(2*Nx)
    f0Noise[:Nx]=g0Noise

    t = np.linspace(0,T,Nt)

    #compute solution
    f = odeint(RHS,f0,t,args=(alpha,beta))
    g = f[:,:Nx] + 1j*f[:,Nx:]

    fNoise = odeint(RHS,f0Noise,t,args=(alpha,beta))
    gNoise = fNoise[:,:Nx] + 1j*fNoise[:,Nx:]

    if display:
        x = np.linspace(0,L,Nx+1)
        x = x[:-1]
        plt.figure()
        plt.contour(x,t,g.real)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of Real(g)')
        plt.show()

    return g,gNoise

def analyze(gA=None,gAN=None,gB=None,gBN=None):
    """
    Question 1.2
    Add input/output as needed

    Discussion: Add discussion here
    """

    if gA == None:
        gA = np.load("gA.npy")
    if gAN == None:
        gAN = np.load("gAN.npy")
    if gB == None:
        gB = np.load("gB.npy")
    if gBN == None:
        gBN = np.load("gBN.npy")

    tstar=200
    T=200
    Nt,Nx=gA.shape
    L=100
    t = np.linspace(50,T,Nt)

    plt.semilogy(t,np.abs(gA[:,100]-gAN[:,100]),label="A")
    plt.semilogy(t,np.abs(gB[:,100]-gBN[:,100]),label="B")
    plt.legend()
    plt.show()

    """w2B, Pxx2B = welch(gB[tstar],return_onesided=False)
    w2A, Pxx2A = welch(gA[tstar],return_onesided=False)

    w2B = w2B*Nx/L*2*np.pi
    w2A = w2A*Nx/L*2*np.pi

    plt.semilogy(np.fft.fftshift(w2B),np.fft.fftshift(Pxx2B),label="B")
    plt.semilogy(np.fft.fftshift(w2A),np.fft.fftshift(Pxx2A),label="A")
    plt.show()

    #plt.plot(gA[200,:-1].imag,gA[200,1:].imag)
    plt.plot(gA[201,:-1].imag,gA[200,1:].imag)
    plt.show()"""

    
    """cB = np.fft.fft(gB)
    cB = np.fft.fftshift(cB)/Nx

    plt.imshow(np.abs(cB), extent=[ -Nx/2, Nx/2, 0, 200])
    plt.show()
    cA = np.fft.fft(gA)
    cA = np.fft.fftshift(cA)/Nx
    plt.imshow(np.abs(cA), extent=[ -Nx/2, Nx/2, 0, 200])
    plt.show()"""
    return None #modify as needed

def getMatrices(Nx,L=100):
    Ab = np.zeros((3,Nx))

    Ab[0, 2:] = 0.375
    Ab[2,:-2] = 0.375
    Ab[1,:] = 1.0

    Ab[0,1] = 3
    Ab[2,-2] = 3

    h = L/Nx
    a = 1.5625 / (2*h)
    b = 0.2 / (4*h)
    c = -0.0125 / (6*h)


    B = sp.lil_matrix((Nx,Nx))
    B.setdiag(c,3)
    B.setdiag(b,2)
    B.setdiag(a,1)
    B.setdiag(-c,-3)
    B.setdiag(-b,-2)
    B.setdiag(-a,-1)

    B[0,0] = -17/6/h
    B[0,1] = 1.5/h
    B[0,2] = 1.5/h
    B[0,3] = -1/6/h
    B[-1,-1] = 17/6/h
    B[-1,-2] = -1.5/h
    B[-1,-3] = -1.5/h
    B[-1,-4] = 1/6/h

    B[2,-1] = -c
    B[1,-2] = -c
    B[1,-1] = -b
    B[-3,0] = c
    B[-2,0] = b
    B[-2,1] = c

    B = B.tocsr()

    return Ab, B


def fdDerivative(g,matrices):
    dgdx = solve_banded((1,1), matrices[0], matrices[1]@g)
    return dgdx


def getK(Nx, L=100):
    n = np.arange(-Nx/2,Nx/2)
    n = np.fft.fftshift(n)
    k = 2j*np.pi*n/L
    return k


def fourierDerivative(g,k):
    c = np.fft.fft(g)
    dgdx = np.fft.ifft(k*c)
    return dgdx

def wavediff(g=None,startSavingAt=1):
    """
    Question 1.3
    Add input/output as needed

    Discussion: Add discussion here
    """
    L=100
    if g == None:
        g = np.load("gLong.npy")
    
    #COMPARE ERRORS
    #---------------------------------------------------------------------

    def getDerivatives(Nx):
        Ab, B = getMatrices(Nx)
        k = getK(Nx)
        dgFiniteDiff = fdDerivative(g[::512//Nx],(Ab,B))
        dgFourier = fourierDerivative(g[::512//Nx],k)
        return dgFiniteDiff, dgFourier

    #compute derivatives
    fdFine, fourierFine = getDerivatives(512)
    fdMed, fourierMed = getDerivatives(256)
    fdCoarse, fourierCoarse = getDerivatives(128)
    

    Nx=512
    x = np.linspace(0,L,Nx+1)
    x = x[:-1]
    plt.semilogy(x,np.abs(fdFine-fourierFine))
    plt.xlabel("x")
    plt.ylabel("Difference between the two methods")
    plt.title("Enrico Ancilotto - Generated by wavediff\nDifference in computation of dg/dx between Fourier and FD methods for fine grid")
    plt.savefig("fig"+str(startSavingAt)+".png", bbox_inches='tight')
    plt.show()
    
    derivativePrecise = (fdFine + fourierFine)/2

    plt.semilogy(x[::4],np.abs(fdCoarse-derivativePrecise[::4]),label="Finite difference, Nx = 128")
    plt.semilogy(x[::4],np.abs(fourierCoarse-derivativePrecise[::4]),label="Fourier, Nx = 128")
    plt.semilogy(x[::2],np.abs(fdMed-derivativePrecise[::2]),label="Finite difference, Nx = 256")
    plt.semilogy(x[::2],np.abs(fourierMed-derivativePrecise[::2]),label="Fourier, Nx = 256")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.title("Enrico Ancilotto - Generated by wavediff\nError in computation of dg/dx for coarser grids")
    plt.savefig("fig"+str(startSavingAt+1)+".png", bbox_inches='tight')
    plt.show()

    plt.semilogy(x[::4],np.abs(fdCoarse-fourierFine[::4]),label="Finite difference, Nx = 128")
    plt.semilogy(x[::4],np.abs(fourierCoarse-fourierFine[::4]),label="Fourier, Nx = 128")
    plt.semilogy(x[::2],np.abs(fdMed-fourierFine[::2]),label="Finite difference, Nx = 256")
    plt.semilogy(x[::2],np.abs(fourierMed-fourierFine[::2]),label="Fourier, Nx = 256")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.title("Enrico Ancilotto - Generated by wavediff\nError in computation of dg/dx for coarser grids using Fourier derivative as reference")
    plt.savefig("fig"+str(startSavingAt+2)+".png", bbox_inches='tight')
    plt.show()

    #FIND EXECUTION TIME
    #--------------------------------------------------------------
    colNames = ["Time","Method","Nx","Setup"]
    dfTimes = pd.DataFrame(columns=colNames)
    Nxs = [32,64,128,256,512]

    def timeF(f,gCopy,param,nLoops):
        """
        Finds average runtime of f (either fd or fourier derivative) with initial setup
        """
        average = 0
        for _ in range(nLoops):
            start = time.time()
            f(gCopy,param)
            average += time.time() - start
        return average/nLoops

    def timeFSetup(f,gCopy,paramFunction,Nx,nLoops):
        """
        Finds average runtime of f (either fd or fourier derivative) without initial setup
        """
        average = 0
        for _ in range(nLoops):
            start = time.time()
            f(gCopy,paramFunction(Nx))
            average += time.time() - start
        return average/nLoops

    nLoops = 10000
    nLoopsSetup = 100
    for Nx in Nxs:
        gCopy = g[::512//Nx]

        #Fourier with setup
        k = getK(Nx)
        t = timeF(fourierDerivative, gCopy, k, nLoops)
        results = {"Time": t, 
                   "Method":"Fourier",
                   "Nx":Nx,
                   "Setup":"Yes"}
        dfTimes.loc[len(dfTimes)] = results 
        
        #FD with setup
        matrices = getMatrices(Nx)
        t = timeF(fdDerivative, gCopy, matrices, nLoops)
        results = {"Time": t, 
                   "Method":"Finite Difference",
                   "Nx":Nx,
                   "Setup":"Yes"}
        dfTimes.loc[len(dfTimes)] = results 

        #Fourier without setup
        t = timeFSetup(fourierDerivative, gCopy, getK, Nx, nLoops)
        results = {"Time": t, 
                   "Method":"Fourier",
                   "Nx":Nx,
                   "Setup":"No"}
        dfTimes.loc[len(dfTimes)] = results 
        
        #FD without setup
        t = timeFSetup(fdDerivative, gCopy, getMatrices, Nx, nLoopsSetup)
        results = {"Time": t, 
                   "Method":"Finite Difference",
                   "Nx":Nx,
                   "Setup":"No"}
        dfTimes.loc[len(dfTimes)] = results 
    fig = sns.catplot("Nx", "Time", hue="Method", col="Setup", data=dfTimes, kind="bar")
    fig.fig.get_axes()[0].set_yscale('log')
    fig.fig.suptitle('Enrico Ancilotto - Generated by wavediff\nTime taken to compute dg/dx')
    plt.savefig("fig"+str(startSavingAt+3)+".png", bbox_inches='tight')
    plt.show()

    return None #modify as needed

def init():
    sns.set()
    if not os.path.isfile("./gLong.npy"):
        g = nwave(1-1j,1+2j,Nx=512,T=100,Nt=401,display=False)
        g = g[-1]
        np.save("gLong.npy",g)

    if not os.path.isfile("./gA.npy"):
        gA,gAN = nwaveNoise(1-2j,1+2j,display=False)
        gA = gA[100:]
        gAN = gAN[100:]
        np.save("gA.npy",gA)
        np.save("gAN.npy",gAN)

    if not os.path.isfile("./gB.npy"):
        g,gN = nwaveNoise(1-1j,1+2j,display=False)
        g = g[100:]
        gN = gN[100:]
        np.save("gB.npy",g)
        np.save("gBN.npy",gN)

if __name__=='__main__':
    x=None
    init()
    #Add code here to call functions above and
    #generate figures you are submitting