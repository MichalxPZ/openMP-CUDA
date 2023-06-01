//
// Created by Michał Zieliński on 01/06/2023.
//

#ifndef PROW_DIJKSTRA_H
#define PROW_DIJKSTRA_H

//
// Created by Michal on 01.06.2023.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <json-c/json.h>
#include "graphGenerator.h"


using namespace std;

void dijkstraSerial(Graph* graph, int source);

void dijkstraOpenMP(Graph* graph, int source);

void dijkstraCUDA(Graph* graph, int source);

Graph loadGraphFromFile(const char* filename);

int dijkstra();


#endif //PROW_DIJKSTRA_H
