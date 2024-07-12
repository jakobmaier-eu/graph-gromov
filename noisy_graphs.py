import numpy as np
import networkx as nx

def alpha_preserving_expected_edges(G, rho):
    return -1

def alpha_preserving_expected_nodes(G, rho):
    return -1

def random_subgraph(G, rho):
    # suppose that G is a simple graph
    # rho = probability of keeping an edge
    if rho < 0 or rho > 1:
        raise ValueError("rho must be in [0,1]")
    Gsmall = G.copy()
    for e in Gsmall.edges:
        if np.random.rand() > rho:
            Gsmall.remove_edge(e)
    return Gsmall, nx.adjacency_matrix(Gsmall)

def noisy_copy_same_nodes(G, rho, alpha="preserve"):
    # suppose that G is a simple graph
    # rho = probability of keeping an edge
    # alpha = probability of adding an edge afterwards
    if alpha == "preserve":
        alpha = alpha_preserving_expected_edges(G, rho)
    if rho < 0 or rho > 1:
        raise ValueError("rho must be in [0,1]")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0,1]")
    # Edge removal procedure
    Gsmall = G.copy()
    for e in Gsmall.edges:
        if np.random.rand() > rho:
            Gsmall.remove_edge(e)
    # Edge addition procedure
    Gprime = Gsmall.copy()
    for i in Gprime.nodes:
        for j in Gprime.nodes:
            if i != j and not Gprime.has_edge(i,j):
                if np.random.rand() < alpha:
                    Gprime.add_edge(i,j)        
    return Gprime, nx.adjacency_matrix(Gprime)




# ignore for no
def noisy_copy_different_nodes(G, rho, alpha="preserve"):
    # suppose that G is a simple graph
    # rho = probability of keeping a node
    # alpha = expected proportion of nodes to add
    if alpha == "preserve":
        alpha = alpha_preserving_expected_nodes(G, rho)
    if rho < 0 or rho > 1:
        raise ValueError("rho must be in [0,1]")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0,1]")
    # Node removal procedure
    n = G.number_of_nodes()
    Gsmall = G.copy()
    for i in Gsmall.nodes:
        if np.random.rand() > rho:
            Gsmall.remove_node(i)
    # Node addition procedure
    nplus = np.random.binomial(n, alpha)
    Gprime = Gsmall.copy()
    for i in range(n, n+nplus):
        Gprime.add_node(i)
    # Connect new nodes with old graph and each other
    for i in Gprime.nodes:
        if i >= n:
            for j in Gprime.nodes:
                if np.random.rand() < 3:
                    Gprime.add_edge(i,j)
        
    return Gprime, nx.adjacency_matrix(Gprime)
