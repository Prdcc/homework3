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

    Discussion: Add discussion here
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

    Discussion: Add discussion here
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
    expI = expB[n:]
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

    Discussion: Add discussion here
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

    zero = np.zeros((n,n))
    eye = np.eye(n)
    B=np.block([[F-(tau+g+k)*eye, a*eye, zero],
                [theta*eye,F-(tau+k+a)*eye,zero],
                [-theta*eye,zero,F-(tau-k)*eye]])

    expB = expm(T*B)

    bSpV = expB[:n]+expB[2*n:]
    bSmV = expB[:n]-expB[2*n:]

    C = bSpV.T @ bSpV - bSmV.T @ bSmV

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

    Discussion: Add discussion here
    """
    if D == None:
        D = np.loadtxt("data.txt")
    M, N = D.shape
    X2 = D - np.outer(np.ones((M,1)),D.mean(axis=0))
    U = np.linalg.svd(X2.T)[0]

    Inew = U[:,0].T @ X2.T
    return Inew


if __name__=='__main__':
    G=None
    #add/modify code here if/as desired
    #N,M = 100,5
    #G = nx.barabasi_albert_graph(N,M,seed=1)