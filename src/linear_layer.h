#pragma once

#include "tensor.h"
#include "layer.h"
#include "opt_alg.h"
#include "probe.h"

struct LinearLayer;
/*
struct LinearLayer
{
    // 基类，接口类
    struct Layer base;

    struct Tensor *w; // n_inputs * n_outputs，布局与yolov2保持一致
    struct Tensor *b;
    struct Tensor *w_grad; // n_inputs * n_outputs，布局与yolov2保持一致
    struct Tensor *b_grad;
};
*/

int createLinearLayer(struct LinearLayer **l, const char *name, int n_in, int n_out);
void destroyLinearLayer(struct LinearLayer *layer);

int getLinearLayerShape(int *n_in, int *n_out, const struct LinearLayer *layer);
int getLinearLayerInputNumber(int *n_in, const struct LinearLayer *layer);
int getLinearLayerOutputNumber(int *n_out, const struct LinearLayer *layer);

int forwardLinearLayer(struct LinearLayer *layer, const struct UpdateArgs *args, struct Probe *probe);
int backwardLinearLayer(struct LinearLayer *layer, const struct UpdateArgs *args, struct Probe *probe);
int updateLinearLayer(struct LinearLayer *layer, const struct UpdateArgs *args, struct Probe *probe);