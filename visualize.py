    import json
    import networkx as nx
    import matplotlib.pyplot as plt

    def visualizeGraph(filename):
        with open(filename, 'r') as file:
            data = json.load(file)

        vertices = data['vertices']
        adjacencyMatrix = data['adjacencyMatrix']

        G = nx.Graph()
        G.add_nodes_from(range(vertices))

        for i in range(vertices):
            for j in range(i + 1, vertices):
                if adjacencyMatrix[i][j] == 1:
                    G.add_edge(i, j)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', edge_color='gray')
        plt.title('Graph Visualization')
        plt.show()

    filename = 'cmake-build-debug/graph_30.json'
    visualizeGraph(filename)