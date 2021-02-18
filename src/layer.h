#pragma once

#include "tensor.h"
#include "opt_alg.h"
#include "probe.h"
#include "const.h"

enum LayerType
{
    UNKNOW_LAYER_TYPE,
    LINEAR_LAYER_TYPE,
    SIGMOID_LAYER_TYPE,
    SOFTMAX_LAYER_TYPE
};

struct Layer
{
    int idx; // Layer对象索引号，对于线形网络就是层号，对于树形或图形网络就是当前层拓扑排序的序号
    char name[NN_LAYER_NAME_LEN];
    enum LayerType type;

    // ref only, memory not own 
    struct Tensor *input;
    struct Tensor *output;
    struct Tensor *delta_in;
    struct Tensor *delta_out;
};

int forwardLayer(struct Layer *layer, const struct UpdateArgs *args, struct Probe *probe);
int backwardLayer(struct Layer *layer, const struct UpdateArgs *args, struct Probe *probe);
int updateLayer(struct Layer *layer, const struct UpdateArgs *args, struct Probe *probe);

int getLayerInputNumber(int *n_in, const struct Layer *layer);
int getLayerOutputNumber(int *n_out, const struct Layer *layer);
int getLayerShape(int *n_in, int *n_out, const struct Layer *layer);

int setLayerName(struct Layer *layer, const char *name);
int setLayerIndex(struct Layer *layer, int idx);
int setLayerInput(struct Layer *layer, const struct Tensor *input);
int setLayerOutput(struct Layer *layer, const struct Tensor *output);
int setLayerInputDelta(struct Layer *layer, const struct Tensor *delta_in);
int setLayerOutputDelta(struct Layer *layer, const struct Tensor *delta_out);