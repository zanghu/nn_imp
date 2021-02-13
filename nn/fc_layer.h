#pragma once

#include "matrix.h"

struct FCLayer;

struct FCLayer *createFCLayer(int n_in, int n_out, const char *act_str);

void destroyFCLayer(struct FCLayer *layer);

int getFCLayerInputNeuronNumber(const struct FCLayer *layer);

int getFCLayerOutputNeuronNumber(const struct FCLayer *layer);

int forwardFCLayer(struct Matrix *output, struct Matrix *hidden, struct FCLayer *layer, const struct Matrix *input);

int backwardFCLayer(struct Matrix *delta_new, const struct Matrix *output, struct FCLayer *layer, const struct Matrix *delta);

int updateFCLayer(struct FCLayer *layer, float lr, float momentum);
