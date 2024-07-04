import networkx as nx
import numpy as np
from scipy.stats import semicircular
from scipy.optimize import root_scalar
from functions import matrix_heatmap

# Every function in here yields adjacency matrices

def sample_Erdos_Renyi(n, p, seed = 123):
    G = nx.erdos_renyi_graph(n, p, seed, directed=False)
    return G, nx.adjacency_matrix(G)

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
