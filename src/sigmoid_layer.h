#pragma once

#include "layer.h"
#include "opt_alg.h"
#include "probe.h"

struct SigmoidLayer;

/*
struct Sigmoid 
{
    struct Layer layer;
    int n_neurons;
};
*/

int createSigmoidLayer(struct SigmoidLayer **l, const char *name, int n_neurons);
void destroySigmoidLayer(struct SigmoidLayer *layer);

int getSigmoidLayerShape(int *n_in, int *n_out, const struct SigmoidLayer *layer);
int getSigmoidLayerInputNumber(int *n_in, const struct SigmoidLayer *layer);
int getSigmoidLayerOutputNumber(int *n_out, const struct SigmoidLayer *layer);

int forwardSigmoidLayer(struct SigmoidLayer *layer, const struct UpdateArgs *args, struct Probe *probe);
int backwardSigmoidLayer(struct SigmoidLayer *layer, const struct UpdateArgs *args, struct Probe *probe);