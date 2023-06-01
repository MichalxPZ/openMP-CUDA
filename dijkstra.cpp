//
// Created by Michal on 01.06.2023.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <json-c/json.h>

#define MAX_VERTICES 100

struct Edge {
    int source;
    int destination;
    int weight;
};

struct Graph {
    int vertices;
    int edges;
    int adjacencyMatrix[MAX_VERTICES][MAX_VERTICES];
};

void convertToEdgeList(Graph* graph, std::vector<Edge>& edgeList) {
    int index = 0;

    for (int i = 0; i < graph->vertices; i++) {
        for (int j = i + 1; j < graph->vertices; j++) {
            if (graph->adjacencyMatrix[i][j] != 0) {
                Edge edge;
                edge.source = i;
                edge.destination = j;
                edge.weight = graph->adjacencyMatrix[i][j];
                edgeList.push_back(edge);
                index++;
            }
        }
    }

    graph->edges = index;
}

void dijkstraSerial(Graph* graph, int source) {
    cout << "serial";
}

void dijkstraOpenMP(Graph* graph, int source) {
    cout << "openmp";
}

void dijkstraCUDA(Graph* graph, int source) {
    cout << "serial";
}

int main() {
    const std::string filename = "graph.json";
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Błąd podczas otwierania pliku." << std::endl;
        return 1;
    }

    // Odczytanie pliku JSON
    std::string buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Parsowanie pliku JSON
    json_object* jsonObj = json_tokener_parse(buffer.c_str());
    json_object* jsonVertices;
    json_object* jsonAdjacencyMatrix;

    Graph graph;
    std::vector<Edge> edgeList;

    // Odczytanie liczby wierzchołków
    json_object_object_get_ex(jsonObj, "vertices", &jsonVertices);
    graph.vertices = json_object_get_int(jsonVertices);

    // Odczytanie macierzy sąsiedztwa
    json_object_object_get_ex(jsonObj, "adjacencyMatrix", &jsonAdjacencyMatrix);
    int i, j;
    json_object* jsonRow;
    json_object* jsonWeight;
    for (i = 0; i < graph.vertices; i++) {
        jsonRow = json_object_array_get_idx(jsonAdjacencyMatrix, i);
        for (j = 0; j < graph.vertices; j++) {
            jsonWeight = json_object_array_get_idx(jsonRow, j);
            graph.adjacencyMatrix[i][j] = json_object_get_int(jsonWeight);
        }
    }

    // Konwersja macierzy sąsiedztwa na listę krawędzi
    convertToEdgeList(&graph, edgeList);

    // Wywołanie algorytmu Dijkstry w różnych wersjach
    int sourceVertex = 0;

    // Algorytm Dijkstry - wersja zwykła (sekwencyjna)
    dijkstraSerial(&graph, sourceVertex);

    // Algorytm Dijkstry - wersja równoległa
    dijkstraOpenMP(&graph, sourceVertex);

    // Algorytm Dijkstry - wersja równoległa (CUDA)
    dijkstraCUDA(&graph, sourceVertex);

    return 0;
}

