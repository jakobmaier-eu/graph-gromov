from frank_wolfe import *
from sample_graphs import *

n_nodes = 20
average_degree = 5
p = average_degree/n_nodes

n_graphs = 5
sampling_function = sample_spherical_geometric 
# ^-- can replace this with sample_spherical_geometric
dataset_graph_objects = []
dataset_adj_matrices = []

for random_seed in range(n_graphs):
    G, A = sampling_function(n_nodes, p, seed=random_seed)
    dataset_graph_objects.append(G)
    dataset_adj_matrices.append(A)

# Drawing a graph (helps debugging):
G = dataset_graph_objects[1]
nx.draw(G)
plt.show()