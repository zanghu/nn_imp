#pragma once

#include "layer.h"
#include "opt_alg.h"
#include "probe.h"

struct SoftmaxLayer;

/*
struct SoftmaxLayer
{
    struct Layer layer;
    int n_neurons;
};
*/

int createSoftmaxLayer(struct SoftmaxLayer **l, int n_neurons);
void destroySoftmaxLayer(struct SoftmaxLayer *layer);

int getSoftmaxLayerShape(int *n_in, int *n_out, const struct SoftmaxLayer *layer);
int getSoftmaxLayerInputNumber(int *n_in, const struct SoftmaxLayer *layer);
int getSoftmaxLayerOutputNumber(int *n_out, const struct SoftmaxLayer *layer);

int forwardSoftmaxLayer(struct SoftmaxLayer *layer, const struct UpdateArgs *args, struct Probe *probe);
int backwardSoftmaxLayer(struct SoftmaxLayer *layer, const struct UpdateArgs *args, struct Probe *probe);