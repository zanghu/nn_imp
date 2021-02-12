#pragma once

#include "matrix.h"

struct Layer;

struct Layer *createLayer(int n_in, int n_out);

void destroyLayer(struct Layer *layer);

int getLayerInputNeuronNumber(const struct Layer *layer);

int getLayerOutputNeuronNumber(const struct Layer *layer);

int forwardLayer(struct Matrix *output, struct Matrix *hidden, const struct Layer *layer, const struct Matrix *input);

int backwardLayer(struct Matrix *delta_new, const struct Matrix *output, const struct Layer *layer, const struct Matrix *delta);

int updateLayer(struct Layer *layer);

