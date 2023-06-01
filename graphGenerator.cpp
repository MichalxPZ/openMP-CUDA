#include "GraphGenerator.h"

void generateRandomGraph(Graph& graph, int density) {
    int vertices = graph.vertices;
    int maxEdges = vertices * (vertices - 1) / 2;
    int edgeCount = density * maxEdges / 100;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(0, vertices - 1);

    int count = 0;
    while (count < edgeCount) {
        int source = distrib(gen);
        int destination = distrib(gen);
        int weight = distrib(gen) + 1; // Losowa waga krawędzi (od 1 do max wartości losowej)

        if (source != destination) {
            graph.edges.push_back({source, destination, weight});
            count++;
        }
    }
}

void saveGraphToFile(const Graph& graph, const char* filename) {
    json_object* jsonObj = json_object_new_object();

    json_object* jsonVertices = json_object_new_int(graph.vertices);
    json_object_object_add(jsonObj, "vertices", jsonVertices);

    json_object* jsonEdges = json_object_new_array();
    for (const auto& edge : graph.edges) {
        json_object* jsonEdge = json_object_new_object();
        json_object* jsonSource = json_object_new_int(edge.source);
        json_object_object_add(jsonEdge, "source", jsonSource);
        json_object* jsonDestination = json_object_new_int(edge.destination);
        json_object_object_add(jsonEdge, "destination", jsonDestination);
        json_object* jsonWeight = json_object_new_int(edge.weight);
        json_object_object_add(jsonEdge, "weight", jsonWeight);
        json_object_array_add(jsonEdges, jsonEdge);
    }
    json_object_object_add(jsonObj, "edges", jsonEdges);

    FILE* file = fopen(filename, "w");
    if (file != nullptr) {
        fprintf(file, "%s", json_object_to_json_string_ext(jsonObj, JSON_C_TO_STRING_PRETTY));
        fclose(file);
        std::cout << "Graf został zapisany do pliku: " << filename << std::endl;
    } else {
        std::cout << "Błąd podczas zapisywania pliku." << std::endl;
    }

    json_object_put(jsonObj);
}

int generate() {
    std::vector<int> graphSizes = {
            10, 25, 50, 75, 100,
            125, 150, 175, 200,
            225, 250, 275, 300,
            325, 350, 375, 400,
            425, 450, 475, 500
    }; // Rozmiary grafów do wygenerowania
    int density = 50;

    for (int size : graphSizes) {
        Graph graph;
        graph.vertices = size;

        generateRandomGraph(graph, density);

        std::string filename = "graph_" + std::to_string(size) + ".json";
        saveGraphToFile(graph, filename.c_str());
    }

    return 0;
}
