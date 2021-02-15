#pragma once

#include "layer.h"

struct SigmoidLayer;

/*
struct Sigmoid 
{
    struct Layer layer;
    int n_neurons;
};
*/

int createSigmoidLayer(struct SigmoidLayer **l, int n_neurons);
void destroySigmoidLayer(struct SigmoidLayer *layer);

int getSigmoidLayerShape(int *n_in, int *n_out, const struct SigmoidLayer *layer);
int getSigmoidLayerInputNumber(int *n_in, const struct SigmoidLayer *layer);
int getSigmoidLayerOutputNumber(int *n_out, const struct SigmoidLayer *layer);

int forwardSigmoidLayer(struct SigmoidLayer *layer);
int backwardSigmoidLayer(struct SigmoidLayer *layer);