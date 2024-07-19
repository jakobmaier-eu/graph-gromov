import networkx as nx
import numpy as np
from scipy.stats import semicircular
from scipy.optimize import root_scalar
from functions import matrix_heatmap

# Every function in here yields adjacency matrices

def sample_Erdos_Renyi(n, p):
    G = nx.erdos_renyi_graph(n, p, directed=False)
    return G, nx.adjacency_matrix(G)

import networkx as nx
import numpy as np

def sample_correlated_Erdos_Renyi(n, p, s):
    """
    Generates two correlated Erdos-Renyi random graphs and their adjacency matrices.
    
    Parameters:
    - n (int): The number of nodes in the graph.
    - p (float): The probability of an edge between any two nodes in the graph.
    - s (float): correlation parameter between the two graphs.

    Returns:
    - G (networkx.Graph): The first Erdos-Renyi random graph.
    - Gp (networkx.Graph): The second Erdos-Renyi random graph.
    - A (scipy.sparse.csr_matrix): The adjacency matrix of G.
    - Ap (scipy.sparse.csr_matrix): The adjacency matrix of Gp.
    """
    G = nx.erdos_renyi_graph(n, p/s, directed=False)
    Gp = G.copy()
    for i in G.nodes:
        for j in G.nodes:
            if i != j:
                if G.has_edge(i,j) and np.random.rand() < 1 - s:
                    G.remove_edge(i,j)
                if Gp.has_edge(i,j) and np.random.rand() < 1- s:
                    Gp.remove_edge(i,j)
    return G, Gp, nx.adjacency_matrix(G), nx.adjacency_matrix(Gp)

def sample_correlated_geometric(n, p, d, s):
    """
    Generate correlated geometric graphs.

    Parameters:
    - n (int): Number of nodes in the graph.
    - p (float): Probability parameter for the semicircular distribution.
    - d (int): Dimension of the random vectors.
    - s (float): Correlation parameter between the two graphs

    Returns:
    - G (nx.Graph): Original graph.
    - Gp (nx.Graph): Graph with added noise.
    - A (scipy.sparse.csr_matrix): Adjacency matrix of the original graph.
    - Ap (scipy.sparse.csr_matrix): Adjacency matrix of the graph with added noise.
    """
    semicirc = semicircular()
    fun = lambda x: semicirc.sf(x) - p
    t = root_scalar(fun, bracket = [-1, 1]).root
    vectors = np.random.randn(d, n)
    vectors_prime = vectors + (1-s) * np.random.randn(d, n)
    vectors = vectors / np.linalg.norm(vectors, axis=0)
    vectors_prime = vectors_prime / np.linalg.norm(vectors_prime, axis=0)
    G = nx.Graph()
    Gp = nx.Graph()
    for i in range(n):
        G.add_node(i)
        Gp.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            if np.dot(vectors[:,i], vectors[:,j]) >= t:
                G.add_edge(i,j)
            if np.dot(vectors_prime[:,i], vectors_prime[:,j]) >= t:
                Gp.add_edge(i,j)
    return G, Gp, nx.adjacency_matrix(G), nx.adjacency_matrix(Gp)

def gaussian_pointcloud_normalised(d, n, seed):
    np.random.seed(seed)
    vectors = np.random.randn(d, n)
    vectors = vectors / np.linalg.norm(vectors, axis=0)
    return vectors 

def sample_spherical_geometric(n, p, d = 3, seed = 123):
    return sample_spherical_geometric_analytical(n, p, d, seed)

def sample_spherical_geometric_analytical(n, p, d, seed):
    semicirc = semicircular()
    fun = lambda x: semicirc.sf(x) - p
    t = root_scalar(fun, bracket = [-1, 1]).root
    vectors = gaussian_pointcloud_normalised(d, n, seed)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            if np.dot(vectors[:,i], vectors[:,j]) >= t:
                G.add_edge(i,j)
    return G, nx.adjacency_matrix(G)

def sample_spherical_geometric_empirical(n, p, d, seed):
    vectors = gaussian_pointcloud_normalised(d, n, seed)
    sps = np.zeros([n, n]) # sps = scalar products
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            sps[i,j] = np.dot(vectors[:, i], vectors[:, j])
    for _ in range(int( p*(n*(n-1))/2 )):
        max_coords = np.unravel_index(np.argmax(sps, axis=None), sps.shape)
        G.add_edge(max_coords[0], max_coords[1])
        sps[max_coords] = -100
    matrix_heatmap(sps)
    return G, nx.adjacency_matrix(G)
