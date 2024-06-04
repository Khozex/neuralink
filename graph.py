import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Carregar os datasets gerados
nodes_df = pd.read_csv('data/social_network_nodes.csv')
edges_df = pd.read_csv('data/social_network_edges.csv')

# Construir o grafo
G = nx.Graph()

# Adicionar nós ao grafo
for _, row in nodes_df.iterrows():
    G.add_node(row['id'], name=row['name'], age=row['age'], gender=row['gender'], followers=row['followers'])

# Adicionar arestas ao grafo
for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'], connection_type=row['connection_type'], strength=row['strength'])

# Função para desenhar o grafo com atributos
def draw_graph(G, pos, node_color, title, color_map=plt.cm.Blues):
    plt.figure(figsize=(12, 12))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500, cmap=color_map)
    edges = nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'name'), font_size=10)

    plt.colorbar(nodes, label=title)
    plt.title(title)
    plt.show()

# Layout para o grafo
pos = nx.spring_layout(G, seed=42)

# Desenhar o grafo inicial
draw_graph(G, pos, 'lightblue', "Grafo de Rede Social")

### Calcular Medidas de Centralidade

# Grau de Centralidade
degree_centrality = nx.degree_centrality(G)
nx.set_node_attributes(G, degree_centrality, 'degree_centrality')
node_color = [G.nodes[node]['degree_centrality'] for node in G.nodes()]
draw_graph(G, pos, node_color, "Grau de Centralidade", plt.cm.Blues)

# Centralidade de Betweenness
betweenness_centrality = nx.betweenness_centrality(G)
nx.set_node_attributes(G, betweenness_centrality, 'betweenness_centrality')
node_color = [G.nodes[node]['betweenness_centrality'] for node in G.nodes()]
draw_graph(G, pos, node_color, "Centralidade de Betweenness", plt.cm.Oranges)

# Centralidade de Closeness
closeness_centrality = nx.closeness_centrality(G)
nx.set_node_attributes(G, closeness_centrality, 'closeness_centrality')
node_color = [G.nodes[node]['closeness_centrality'] for node in G.nodes()]
draw_graph(G, pos, node_color, "Centralidade de Closeness", plt.cm.Greens)

### Detecção de Comunidades

# Algoritmo de Louvain
import community as community_louvain

partition = community_louvain.best_partition(G)
nx.set_node_attributes(G, partition, 'community')

# Visualizar comunidades
community_color = [partition[node] for node in G.nodes()]
draw_graph(G, pos, community_color, "Comunidades (Algoritmo de Louvain)", plt.cm.tab20)
