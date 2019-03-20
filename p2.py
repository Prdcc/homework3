"""M345SC Homework 3, part 2
Enrico Ancilotto
01210716
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import svds,expm


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

    B = sp.block_diag((F,F,F), format="lil")
    del F

    #The tau term is given by sum F_ji = tau
    B.setdiag(np.concatenate((np.zeros(n)-tau-g-k, np.zeros(n)-tau-k-a, np.zeros(n)-tau+k),axis=None))
    B.setdiag([a]*n, n)
    B.setdiag([theta]*n,-n)
    B.setdiag([-theta]*n,-2*n)
    
    B.tocsc()

    expB = expm(T*B)
    _, s, vt = svds(expB.tocsc(), k=1, return_singular_vectors="vh")
    s = s[0]
    vt = vt[0]
    
    growth = s**2
    return growth, vt

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

    B = sp.block_diag((F,F), format="lil")
    del F

    #The tau term is given by sum F_ji = tau
    B.setdiag(np.concatenate((np.zeros(n)-tau-g-k, np.zeros(n)-tau-k-a, np.zeros(n)-tau+k),axis=None))
    B.setdiag([a]*n, n)
    B.setdiag([theta]*n,-n)

    expB = expm(T*B)
    expB = expB[n:]
    _, s, vt = svds(expB, k=1, return_singular_vectors="vh")
    s = s[0]
    vt = vt[0]
    growth = s**2
    vt = np.append(vt,[0]*n)
    return growth, vt


def growth3(G,params=(2,2.8,0,1,1.0,0.5),T=6):
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
    N = G.number_of_nodes()

    #Construct flux matrix (use/modify as needed)
    Q = [t[1] for t in G.degree()]

    Pden = np.zeros(N)
    Pden_total = np.zeros(N)
    for j in G.nodes():
        for m in G.adj[j].keys():
            Pden[j] += Q[m]
    Pden = 1/Pden
    Q = tau*np.array(Q)
    F = nx.adjacency_matrix(G).toarray()
    F = F*np.outer(Q,Pden)
    #-------------------------------------
    G=0

    #Add code here

    return G


def Inew(D):
    """
    Question 2.4

    Input:
    D: N x M array, each column contains I for an N-node network

    Output:
    I: N-element array, approximation to D containing "large-variance"
    behavior

    Discussion: Add discussion here
    """
    # N,M = D.shape
    # I = np.zeros(N)

    #Add code here


    return I


if __name__=='__main__':
    G=None
    #add/modify code here if/as desired
    #N,M = 100,5
    #G = nx.barabasi_albert_graph(N,M,seed=1)