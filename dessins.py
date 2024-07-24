import networkx as nx
import matplotlib.pyplot as plt

# Création des deux graphes
G1 = nx.Graph()
G2 = nx.Graph()

# Ajout des nœuds et des arêtes pour le premier graphe
G1.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])

# Ajout des nœuds et des arêtes pour le second graphe
G2.add_edges_from([("A'", "B'"), ("A'", "C'"), ("B'", "D'"), ("C'", "D'")])

# Positionnement des nœuds pour les deux graphes
pos_G1 = {'A': (0, 3), 'B': (2, 4), 'C': (2, 2), 'D': (4, 3)}
pos_G2 = {"A'": (7, 3), "B'": (9, 4), "C'": (9, 2), "D'": (11, 3)}

# Création d'un nouveau graphe pour inclure les alignements
G_combined = nx.Graph()

# Ajouter les nœuds et les arêtes des deux graphes dans G_combined
G_combined.add_nodes_from(G1.nodes(data=True))
G_combined.add_nodes_from(G2.nodes(data=True))
G_combined.add_edges_from(G1.edges(data=True))
G_combined.add_edges_from(G2.edges(data=True))

# Ajout des alignements (arêtes pointillées entre les deux graphes)
alignment_edges = [('A', "A'"), ('B', "B'"), ('C', "C'"), ('D', "D'")]
G_combined.add_edges_from(alignment_edges)

# Positionnement combiné des deux graphes
pos_combined = {**pos_G1, **pos_G2}

# Dessin des graphes
plt.figure(figsize=(10, 5))

# Dessiner les nœuds et les arêtes des deux graphes
nx.draw(G_combined, pos_combined, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', edge_color='black')

# Dessiner les alignements en pointillés
nx.draw_networkx_edges(G_combined, pos_combined, edgelist=alignment_edges, style='dashed', edge_color='red')

plt.show()
