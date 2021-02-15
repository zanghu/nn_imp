#pragma once

#include "tensor.h"

enum CostType
{
    UNKNOW_COST_TYPE,
    CE_COST_TYPE
};

struct Cost
{
    enum CostType type;
    int n_input;
    float value; // 代价值

    // ref
    struct Tensor *input;
    struct Tensor *delta;
};

int getCostInputNumber(int *n_in, const struct Cost *cost);
int getCostValue(float *val, const struct Cost *cost);
int setCostInput(struct Cost *cost, const struct Tensor *input);
int setCostDelta(struct Cost *cost, const struct Tensor *delta);

int forwardCost(struct Cost *cost);
int backwardCost(struct Cost *cost, const struct Tensor *gt);