//
// Created by Michał Zieliński on 01/06/2023.
//

#ifndef PROW_GRAPHGENERATOR_H
#define PROW_GRAPHGENERATOR_H
#include <iostream>
#include <vector>
#include <random>
#include <json-c/json.h>

#define MAX_VERTICES 500

struct Edge {
    int source;
    int destination;
    int weight;
};
struct Graph {
    int vertices;
    std::vector<Edge> edges;
};
void generateRandomGraph(Graph& graph, int density);

void saveGraphToFile(const Graph& graph, const char* filename);

int generate();

#endif //PROW_GRAPHGENERATOR_H
