#pragma once

#include "tensor.h"
#include "layer.h"
#include "cost.h"
#include "opt_alg.h"

struct Network;

int createNetwork(struct Network **network, struct Layer **layers, int n_layers, struct Cost *cost, const struct UpdateArgs *args);
void destroyNetwork(struct Network *net);

int forwardNetwork(struct Network *net, const struct Tensor *input);
int backwardNetwork(struct Network *net, const struct Tensor *gt);
int updateNetwork(struct Network *net);