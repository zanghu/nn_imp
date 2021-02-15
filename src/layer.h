#pragma once

#include "tensor.h"
#include "opt_alg.h"

enum LayerType
{
    UNKNOW_LAYER_TYPE,
    LINEAR_LAYER_TYPE,
    SIGMOID_LAYER_TYPE,
    SOFTMAX_LAYER_TYPE
};

struct Layer
{
    enum LayerType type;

    // ref only, memory not own 
    struct Tensor *input;
    struct Tensor *output;
    struct Tensor *delta_in;
    struct Tensor *delta_out;

    //float lr;
    //float momentum;
};

int forwardLayer(struct Layer *layer);
int backwardLayer(struct Layer *layer);
int updateLayer(struct Layer *layer, const struct UpdateArgs *args);

int getLayerInputNumber(int *n_in, const struct Layer *layer);
int getLayerOutputNumber(int *n_out, const struct Layer *layer);
int getLayerShape(int *n_in, int *n_out, const struct Layer *layer);

int setLayerInput(struct Layer *layer, const struct Tensor *input);
int setLayerOutput(struct Layer *layer, const struct Tensor *output);
int setLayerInputDelta(struct Layer *layer, const struct Tensor *delta_in);
int setLayerOutputDelta(struct Layer *layer, const struct Tensor *delta_out);