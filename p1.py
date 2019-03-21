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
from scipy.spatial.distance import cdist
import os   #used to check if data has already been computed
import time
import seaborn as sns   #plotting
import pandas as pd

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
    L = 100
    x = np.linspace(0,L,Nx+1)
    x = x[:-1]

    n = np.arange(-Nx/2,Nx/2)
    n = np.fft.fftshift(n)
    k = -(2*np.pi*n/L)**2   #coefficients for computation of second derivative
    
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
        plt.figure()
        plt.contour(x,t,g.real)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of Real(g)')
        plt.show()

    return g

def nwaveNoise(alpha,beta,Nx=256,Nt=801,T=200):
    """
    Same as nwave but also produces a solution computed by adding some noise to 
    the initial conditions.
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

    return g,gNoise

def analyze(gA=None,gAN=None,gB=None,gBN=None,startSavingAt=5):
    """
    Question 1.2
    Add input/output as needed

    Discussion: Figures 4 and 5 show the energy distribution of the various 
    frequencies for both A and B (note that the displayed value is the logarithm 
    of the energy for a certain frequency). As g is a complex value we can't 
    simply discard the negative frequencies, however both graphs are mostly 
    symmetric suggesting that the complex part isn't a crucial part of the 
    signal. The energy level decays exponentially in both the positive and 
    negative direction: most of the energy is concentrated in the lowest 
    frequency waves. 

    Figures 7 through 9 analyse the chaotic behaviour of the two cases. Figure 7 
    plots the correlation dimension for both of them at two separate points in 
    space. We get a fractal dimension of more than one in both cases indicating 
    a chaotic behaviour since we are working with lines in 2D space (the two 
    dimensions are given by real(g) and im(g)). However both times, the slope 
    for A is slightly bigger than the one for B, indicating that the former is 
    slightly more chaotic.

    This finding is confirmed in figure 9 which shows the orbit diagram for the 
    two cases. The orbits do cross for both, but as g(t+Delta t,x) depends not 
    only on g(t,x) but also on g(t, x+Delta' x) this does not prevent aperiodic 
    behaviour. But the orbit in case A is much less condensed around the main 
    circle as would be expected for periodic data, so once again we see a 
    slightly less chaotic behaviour for case B.

    Finally in figure 8 we look at the difference that some noise makes for the 
    two cases. In case B we can see exponential growth between t=25 and 40 
    before the maximum difference is reached. A reaches this maximum level 
    already at t = 30 (most of the exponential growth happens before t=25) 
    suggesting a quicker divergence, and so once again, more chaotic behaviour.
    """
    count = 0   #number of figures saved
    if gA is None:  #load from file if not given as parameter
        gA = np.load("gA.npy")
    if gAN is None:
        gAN = np.load("gAN.npy")
    if gB is None:
        gB = np.load("gB.npy")
    if gBN is None:
        gBN = np.load("gBN.npy")

    tstar=200   #parameters
    xstar = 64
    T=200
    Tstart = 25
    Nt,Nx=gA.shape
    L=100
    t = np.linspace(Tstart,T,Nt)

    #---------------------------------------------------------------------------
    #FIND ENERGY DISTRIBUTION
    #---------------------------------------------------------------------------
    w2B, Pxx2B = welch(gB,return_onesided=False)
    w2A, Pxx2A = welch(gA,return_onesided=False)

    w2B = np.fft.fftshift(w2B*Nx/L*2*np.pi)
    w2A = np.fft.fftshift(w2A*Nx/L*2*np.pi)

    sns.heatmap(xticklabels=False,yticklabels=False,data=np.fft.fftshift(np.log(Pxx2A)))
    plt.xticks(np.arange(len(w2A))[::20],np.round(w2A[::20],1))
    plt.yticks(np.arange(701)[::-50],np.round(t[::50]))
    plt.xlabel("Frequency")
    plt.ylabel("Time")
    plt.title("Enrico Ancilotto - Generated by analyze\nEnergy at different frequencies for case A, log-scale")
    plt.savefig("fig"+str(startSavingAt+count)+".png", bbox_inches='tight')
    count+=1
    plt.show()

    sns.heatmap(xticklabels=False,yticklabels=False,data=np.fft.fftshift(np.log(Pxx2B)))
    plt.xticks(np.arange(len(w2B))[::20],np.round(w2B[::20],1))
    plt.yticks(np.arange(701)[::-50],np.round(t[::50]))
    plt.xlabel("Frequency")
    plt.ylabel("Time")
    plt.title("Enrico Ancilotto - Generated by analyze\nEnergy at different frequencies for case B, log-scale")
    plt.savefig("fig"+str(startSavingAt+count)+".png", bbox_inches='tight')
    count+=1
    plt.show()


    #---------------------------------------------------------------------------
    #Find correlation dimension
    #---------------------------------------------------------------------------
    def getCorrelationDimension(start,stop,g,x):
        y1 = np.zeros((Nt-1,2))
        y1[:,0] = g[:-1,x].real
        y1[:,1] = g[:-1,x].imag
        
        y2 = np.zeros((Nt-1,2))
        y2[:,0] = g[1:,x].real
        y2[:,1] = g[1:,x].imag

        n=Nt-1

        D = cdist(y1,y2)    #find distances

        epsilon = np.logspace(-2.5,1)
        C = np.zeros(len(epsilon))

        for i in range(len(epsilon)):
            C[i] = D[D<epsilon[i]].size
        C /= n*(n-1)/2  #normalise

        coeff=np.polyfit(epsilon[start:stop],C[start:stop],1)
        return coeff, C

    epsilon = np.logspace(-2.5,1)

    f, axes = plt.subplots(1, 2, sharey=True)
    axes = axes.flatten()
    ax1=axes[0]
    ax2=axes[1]
    
    #x=50
    coeff,C = getCorrelationDimension(30,-12,gB,128)
    ax1.loglog(epsilon,C,label="C(epsilon) for B")
    ax1.loglog(epsilon[30:-12],np.polyval(coeff,epsilon[30:-12]),'k--',label="Best fit line for correlation of B, slope={:3.2f}".format(coeff[0]))

    coeff,C = getCorrelationDimension(27,-15,gA,128)
    ax1.loglog(epsilon,C,label="C(epsilon) for A")
    ax1.loglog(epsilon[27:-15],np.polyval(coeff,epsilon[27:-15]),'r--',label="Best fit line for correlation of A, slope={:3.2f}".format(coeff[0]))
    ax1.set(title="x=50")

    #x=25
    coeff,C = getCorrelationDimension(30,-12,gB,64)
    ax2.loglog(epsilon,C,label="C(epsilon) for B")
    ax2.loglog(epsilon[30:-12],np.polyval(coeff,epsilon[30:-12]),'k--',label="Best fit line for correlation of B, slope={:3.2f}".format(coeff[0]))

    coeff,C = getCorrelationDimension(27,-15,gA,64)
    ax2.loglog(epsilon,C,label="C(epsilon) for A")
    ax2.loglog(epsilon[27:-15],np.polyval(coeff,epsilon[27:-15]),'r--',label="Best fit line for correlation of A, slope={:3.2f}".format(coeff[0]))
    ax2.set(title="x=25")

    for ax in axes: #set labels
        ax.set(xlabel='Epsilon',ylabel="C")
        ax.legend()
    for ax in axes:
        ax.label_outer()

    f.suptitle("Enrico Ancilotto - Generated by analyze\nCorrelation dimension for the two cases at different values of x")
    plt.savefig("fig"+str(startSavingAt+count)+".png", bbox_inches='tight')
    count+=1
    plt.show()


    #---------------------------------------------------------------------------
    #EFFECT OF NOISE
    #---------------------------------------------------------------------------

    plt.semilogy(t,np.abs(gA[:,xstar]-gAN[:,xstar]),label="A")
    plt.semilogy(t,np.abs(gB[:,xstar]-gBN[:,xstar]),label="B")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Difference")
    plt.title("Enrico Ancilotto - Generated by analyze\nDifference between simulation results with and without added noise")
    plt.savefig("fig"+str(startSavingAt+count)+".png", bbox_inches='tight')
    count+=1
    plt.show()

    #---------------------------------------------------------------------------
    #ORBIT DIAGRAMS
    #---------------------------------------------------------------------------

    f, axes = plt.subplots(1, 2, sharey=True)
    axes = axes.flatten()
    axes[0].plot(gA[:,xstar].real, gA[:,xstar].imag)
    axes[0].set(title="Case A")
    axes[1].plot(gB[:,xstar].real, gB[:,xstar].imag)
    axes[1].set(title="Case B")

    for ax in axes:
        ax.set(xlabel='Real(g)',ylabel="Im(g)")
    for ax in axes:
        ax.label_outer()
    f.suptitle("Enrico Ancilotto - Generated by analyze\nOrbit diagram for g at x = 50")
    plt.savefig("fig"+str(startSavingAt+count)+".png", bbox_inches='tight')
    count+=1
    plt.show()
    return None #modify as needed

def getMatrices(Nx,L=100):
    """
    Get matrices for finite differences calculations
    """
    Ab = np.zeros((3,Nx))

    #Left matrix stored in format for sp.solve_banded
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
    """
    Compute fdDerivative of g
    """
    dgdx = solve_banded((1,1), matrices[0], matrices[1]@g)
    return dgdx


def getK(Nx, L=100):
    """
    Returns the coefficients for Fourier differentiation
    """
    n = np.arange(-Nx/2,Nx/2)
    n = np.fft.fftshift(n)
    k = 2j*np.pi*n/L
    return k


def fourierDerivative(g,k):
    """
    Computes derivative of g using Fourier differentiation
    """
    c = np.fft.fft(g)
    dgdx = np.fft.ifft(k*c)
    return dgdx

def wavediff(g=None,startSavingAt=1):
    """
    Question 1.3
    Add input/output as needed

    Discussion:
    In figure 1 we look at the difference between the computed derivatives for 
    the two methods using a very fine grid (Nx=512). The two methods give a very 
    similar answer in the centre of the interval, however we get a large spike 
    in the error at the edges; this is to be expected since the finite 
    difference method should be less accurate there. 

    This is actually confirmed in the next figure: a coarser grid is used for g 
    (either 128 or 256) and we compare the value of dg/dx for the two methods 
    with the average between the precise derivatives computed for figure 1. 
    Using this method we can only be as precise as in the first case, in 
    particular we can't hope for a precision of more than 10^-4 at the edges, 
    both the finite difference computations see a spike bigger than that, while 
    none of the Fourier do, staying in line with the behaviour in the centre. 
    Thus the Fourier derivative will give us a more consistent accuracy. This is 
    used for figure 3, instead of comparing to the average of the two 
    derivatives, we only compare the coarser derivatives to the finer Fourier 
    one. We can clearly see that the two give us about the same precision in the 
    centre with the error of the finite much less consistent. However the 
    Fourier method manages to maintain the same error on the edge, while finite 
    differences see a very big spike. It is also interesting to note that the 
    graphs for the finite difference methods have very similar behaviour (the 
    peaks and troughs align), albeit with the finer one giving a much smaller 
    error.

    In figure 4 we also consider the execution time needed for the two methods. 
    They both need a setup, in the Fourier case we need to compute the value of 
    k, while for finite differences we need the two matrices. These are quite 
    time-consuming operations, however in most cases they can be precomputed and 
    so might not need to be considered if the differentiation is to be repeated. 

    In both cases Fourier differentiation is much quicker, especially when no 
    setup is allowed (admittedly the creation of the matrices might not have 
    been implemented in the most efficient of ways). This seems to contradict 
    the asymptotic time complexity calculations: O(NlogN) for FFT and 
    O(N*i + N*j) for finite difference methods (here i is the number of non-zero 
    diagonals in the left matrix which needs to be inverted and j the same value 
    for the right matrix which has to be multiplied by a vector), since i,j << N 
    we get O(N) in this case. However asymptotic run time analysis is exactly 
    that: asymptotic. It doesn't tell us the behaviour for small N, and clearly 
    even N=512 is too small for the logN term to make a difference. Thus we 
    probably have much smaller coefficients for the Fourier case meaning that in 
    practice it will be more useful.

    All things considered, Fourier differentiation gives a faster, more reliable 
    way to differentiate functions that doesn't give huge errors on the boundary.
    Its main drawback is its limitations to periodic functions that vanish on
    the boundary. But if this is the case it should always be preferred.
    """
    L=100
    if g is None:   #If g isn't set load it from file
        g = np.load("gLong.npy")
    
    #---------------------------------------------------------------------------
    #COMPARE ERRORS
    #---------------------------------------------------------------------------

    def getDerivatives(Nx):
        """
        Returns Fourier and FD derivatives for a given coarseness
        """
        Ab, B = getMatrices(Nx)
        k = getK(Nx)
        dgFiniteDiff = fdDerivative(g[::512//Nx],(Ab,B))
        dgFourier = fourierDerivative(g[::512//Nx],k)
        return dgFiniteDiff, dgFourier

    #compute derivatives
    fdFine, fourierFine = getDerivatives(512)
    fdMed, fourierMed = getDerivatives(256)
    fdCoarse, fourierCoarse = getDerivatives(128)
    
    #plot difference between precise FD and Fourier derivatives
    Nx=512
    x = np.linspace(0,L,Nx+1)
    x = x[:-1]
    plt.semilogy(x,np.abs(fdFine-fourierFine))
    plt.xlabel("x")
    plt.ylabel("Difference between the two methods")
    plt.title("Enrico Ancilotto - Generated by wavediff\nDifference in computation of dg/dx between Fourier and FD methods for fine grid")
    plt.savefig("fig"+str(startSavingAt)+".png", bbox_inches='tight')
    plt.show()
    
    derivativePrecise = (fdFine + fourierFine)/2    #average of two precise derivatives
    
    #Plot difference between coarser and precise derivatives
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

    #Plot difference between coarser and precise Fourier derivatives
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

    #---------------------------------------------------------------------------
    #FIND EXECUTION TIME
    #---------------------------------------------------------------------------
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
        dfTimes.loc[len(dfTimes)] = results #append results
        
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
    
    #plot results
    fig = sns.catplot("Nx", "Time", hue="Method", col="Setup", data=dfTimes, kind="bar")
    fig.fig.get_axes()[0].set_yscale('log')
    fig.fig.suptitle('Enrico Ancilotto - Generated by wavediff\nTime taken to compute dg/dx')
    plt.savefig("fig"+str(startSavingAt+3)+".png", bbox_inches='tight')
    plt.show()

    return None #modify as needed

def init(save):
    sns.set()

    #Compute solutions if not already available, this will take a while (a bit less than 5 minutes for me)
    if not os.path.isfile("./gLong.npy"):
        print("Computing solutions, this may take a while")
        print("Progress 0/3",end="")
        g = nwave(1-1j,1+2j,Nx=512,T=100,Nt=401,display=False)
        g = g[-1]
        print("\rProgress 1/3",end="")
    else:
        g = np.load("gLong.npy")       

    if not os.path.isfile("./gA.npy"):
        gA,gAN = nwaveNoise(1-2j,1+2j)
        gA = gA[100:]
        gAN = gAN[100:]
        print("\rProgress 2/3",end="")
    else:
        gA = np.load("gA.npy")
        gAN = np.load("gAN.npy")

    if not os.path.isfile("./gB.npy"):
        gB,gBN = nwaveNoise(1-1j,1+2j)
        gB = gB[100:]
        gBN = gBN[100:]
        print("\rProgress 3/3",end="")    
    else:
        gB = np.load("gB.npy")
        gBN = np.load("gBN.npy")

    if save:
        np.save("gB.npy",gB)
        np.save("gBN.npy",gBN)
        np.save("gA.npy",gA)
        np.save("gAN.npy",gAN)
        np.save("gLong.npy",g)
    return g, gA, gAN, gB, gBN

if __name__=='__main__':
    x=None
    gLong, gA, gAN, gB, gBN = init(save=True)    #change to true if expecting to execute more than once and don't mind clutter of a few extra files
    wavediff(gLong)
    analyze(gA, gAN, gB, gBN)
