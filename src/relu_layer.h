#pragma once

#include "layer.h"
#include "opt_alg.h"
#include "probe.h"

struct ReluLayer;

int createReluLayer(struct ReluLayer **l, const char *name);
void destroyReluLayer(struct ReluLayer *layer);

int getReluLayerShape(int *n_in, int *n_out, const struct ReluLayer *layer);
int getReluLayerInputNumber(int *n_in, const struct ReluLayer *layer);
int getReluLayerOutputNumber(int *n_out, const struct ReluLayer *layer);
int setReluLayerNeuronNumber(struct ReluLayer *layer, int n_neurons);

int forwardReluLayer(struct ReluLayer *layer, const struct UpdateArgs *args, struct Probe *probe);
int backwardReluLayer(struct ReluLayer *layer, const struct UpdateArgs *args, struct Probe *probe);