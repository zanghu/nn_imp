#pragma once

#include "matrix.h"
#include "fc_layer.h"

struct Network;

struct Network *createNetwork(struct FCLayer **layers, int n_layers, int batch_size, float lr, float momentum);

int resetNetworkToTrain(struct Network *net, int batch_size, float lr, float momentum);
 
int forwardNetwork(struct Network *net, const struct Matrix *input);

int backwardNetwork(struct Network *net, const struct Matrix *delta);

int updateNetwork(struct Network *net);