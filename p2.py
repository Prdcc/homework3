"""M345SC Homework 3, part 2
Enrico Ancilotto
01210716
"""
import numpy as np
import networkx as nx
from scipy.linalg import expm,svd


def growth1(G,params=(0.02,6,1,0.1,0.1),T=6):
    """
    Question 2.1
    Find maximum possible growth, G=e(t=T)/e(t=0) and corresponding initial
    condition.

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth

    Output:
    G: Maximum growth
    y: 3N-element array containing computed initial condition where N is the
    number of nodes in G and y[:N], y[N:2*N], y[2*N:] correspond to S,I,V

    Discussion: The ODE is a linear system, we therefore know that the exact
    solutions is y(T)=exp(BT)y0, where B is the matrix defining the system. Thus
    we can use the theory developed in class to get the desired answer.
    """
    a,theta,g,k,tau=params
    n = G.number_of_nodes()

    #Construct flux matrix (use/modify as needed)
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(n)
    Pden_total = np.zeros(n)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1/Pden
    Q = tau*np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F*np.outer(Q,Pden)
    #-------------------------------------
    del G

    #construct matrix representing linear system
    zero = np.zeros((n,n))
    eye = np.eye(n)
    B=np.block([[F-(tau+g+k)*eye, a*eye, zero],
                [theta*eye,F-(tau+k+a)*eye,zero],
                [-theta*eye,zero,F-(tau-k)*eye]])

    expB = expm(T*B)
    _, s, vt = svd(expB)
    return s[0]**2, vt[0]

def growth2(G,params=(0.02,6,1,0.1,0.1),T=6):
    """
    Question 2.2
    Find maximum possible growth, G=sum(Ii^2)/e(t=0) and corresponding initial
    condition.

    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth

    Output:
    G: Maximum growth
    y: 3N-element array containing computed initial condition where N is the
    number of nodes in G and y[:N], y[N:2*N], y[2*N:] correspond to S,I,V

    Discussion: As before the exact solution will be given by y=exp(BT)y0. As
    we are only interested in I we modify this to get I=exp(BT)[n:2n]. 
    Furthermore, I only depends on I and S, same as S, so it isn't affected by
    V, this means we can set V(0)=0. We can further simplify the problem by only
    considering the top left section of B, as the others describe the interaction
    with V, greatly reducing the number of calculations required.
    """
    a,theta,g,k,tau=params
    n = G.number_of_nodes()

    #Construct flux matrix (use/modify as needed)
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(n)
    Pden_total = np.zeros(n)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1/Pden
    Q = tau*np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F*np.outer(Q,Pden)
    #-------------------------------------
    del G

    eye = np.eye(n)
    B=np.block([[F-(tau+g+k)*eye, a*eye],
                [theta*eye,F-(tau+k+a)*eye]])

    expB = expm(T*B)
    expI = expB[n:]     #expI y0 = I(T)
    _, s, vt = svd(expI) 
    return s[0]**2, np.concatenate((vt[0],[0]*n))


def growth3(G,params=(2,2.8,1,1.0,0.5),T=6):
    """
    Question 2.3
    Find maximum possible growth, G=sum(Si Vi)/e(t=0)
    Input:
    G: Networkx graph
    params: contains model parameters, see code below.
    T: time for energy growth

    Output:
    G: Maximum growth

    Discussion: First note that 4sum SV=||S+V||^2-||S-V||^2. As S+V and S-V are 
    both linear combinations of S,I,V, which are themselves a linear transformation
    of S(0), I(0), V(0) we can express S+V = Bp y0, S-V = Bm y0. ||x||^2=x^Tx, so:
    4sum SV = (y0^T Bp^T Bp y0) - (y0^T Bm^T Bm y0) = y0^T(Bp^T Bp -  Bm^T Bm)y0
    = y0^T C y0. Where C is symmetric, therefore we can apply once again the 
    theory from lectures to say that the maximum growth will be given by the
    biggest eigenvalue of C divided by 4.
    """
    a,theta,g,k,tau=params
    n = G.number_of_nodes()

    #Construct flux matrix (use/modify as needed)
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(n)
    Pden_total = np.zeros(n)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1/Pden
    Q = tau*np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F*np.outer(Q,Pden)
    #-------------------------------------
    del G

    #construct the matrix
    zero = np.zeros((n,n))
    eye = np.eye(n)
    B=np.block([[F-(tau+g+k)*eye, a*eye, zero],
                [theta*eye,F-(tau+k+a)*eye,zero],
                [-theta*eye,zero,F-(tau-k)*eye]])

    expB = expm(T*B)    #compute matrix for solution

    bSpV = expB[:n]+expB[2*n:]  #bSpV y0 = (S+V)(T)
    bSmV = expB[:n]-expB[2*n:]

    C = bSpV.T @ bSpV - bSmV.T @ bSmV   #see discussion for meaning of matrix

    eigenVals = np.linalg.eigvals(C) 

    return max(eigenVals) / 4


def Inew(D=None):
    """
    Question 2.4

    Input:
    D: N x M array, each column contains I for an N-node network

    Output:
    I: N-element array, approximation to D containing "large-variance"
    behavior

    Discussion: This is a straightforward application of PCA. The msot important
    component will be identified by the first eigenvector, so we simply project
    our observation onto it to identify the most important nodes, while best 
    approximating the total variation. We first however have to normalise the data
    so that the mean is zero.
    """
    if D == None:
        D = np.loadtxt("data.txt")
    N, M = D.shape
    X2 = D - np.outer(np.ones((N,1)),D.mean(axis=0))    #normalise data
    U = np.linalg.svd(X2.T)[0]  #find eigenvectors

    Inew = U[:,0].T @ X2.T  #project data
    return Inew


if __name__=='__main__':
    G=None
    #add/modify code here if/as desired
    #N,M = 100,5
    #G = nx.barabasi_albert_graph(N,M,seed=1)