#pragma once

#include "matrix.h"
#include "layer.h"

struct Network;

struct Network *createNetwork(struct Layer **layers, int n_layers, int batch_size);

int resetNetworkToTrain(struct Network *net, int batch_size);
 
int forwardNetwork(struct Network *net, const struct Matrix *input);

int backwardNetwork(struct Network *net, const struct Matrix *delta);

int updateNetwork(struct Network *net);