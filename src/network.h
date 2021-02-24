#pragma once

//#include "tensor.h"
#include "layer.h"
#include "cost.h"
#include "opt_alg.h"
#include "probe.h"

struct Network;

int createNetwork(struct Network **network, struct Layer **layers, int n_layers, struct Cost *cost);
void destroyNetwork(struct Network *net);
int getNetworkClassProbabilityConstRef(const float *(*p), const struct Network *net);
int forwardNetwork(struct Network *net, const void *input_data, int n_samples, int n_features, const char *dtype_str, const struct UpdateArgs *args, struct Probe *probe);
int backwardNetwork(struct Network *net, const void *gt_data, int n_samples, int n_features, const char *dtype_str, const struct UpdateArgs *args, struct Probe *probe);
int updateNetwork(struct Network *net, const struct UpdateArgs *args, struct Probe *probe);