"""M345SC Homework 3, part 1
Enrico Ancilotto
01210716
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import hann

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
    x = np.linspace(0,L,Nx+1)
    x = x[:-1]
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
        plt.figure()
        plt.contour(x,t,g.real)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of Real(g)')
        plt.show()

    return g

def analyze():
    """
    Question 1.2
    Add input/output as needed

    Discussion: Add discussion here
    """

    return None #modify as needed


def wavediff():
    """
    Question 1.3
    Add input/output as needed

    Discussion: Add discussion here
    """

    return None #modify as needed

if __name__=='__main__':
    x=None
    #Add code here to call functions above and
    #generate figures you are submitting