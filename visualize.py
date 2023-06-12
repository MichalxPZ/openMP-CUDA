import json
import networkx as nx
import matplotlib.pyplot as plt

def visualizeGraph(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    edges = data

    G = nx.Graph()

    for edge in edges:
        source = edge['source']
        destination = edge['destination']
        weight = edge['weight']
        G.add_edge(source, destination, weight=weight)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', edge_color='gray')
    plt.title('Graph Visualization')
    plt.show()

filename = 'cmake-build-debug/graph.json'
visualizeGraph(filename)
