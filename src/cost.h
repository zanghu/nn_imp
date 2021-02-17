#pragma once

#include "tensor.h"
#include "probe.h"
#include "opt_alg.h"

enum CostType
{
    UNKNOW_COST_TYPE,
    CE_COST_TYPE
};

struct Cost
{
    char name[64];
    enum CostType type;
    int n_input;
    float value; // 代价值

    // ref
    struct Tensor *input;
    struct Tensor *delta;
};

int getCostInputNumber(int *n_in, const struct Cost *cost);
int getCostValue(float *val, const struct Cost *cost);
int getCostGroundTruthAttributes(int *n_features, enum DType *dtype, const struct Cost *cost);
int setCostInput(struct Cost *cost, const struct Tensor *input);
int setCostDelta(struct Cost *cost, const struct Tensor *delta);

int forwardCost(struct Cost *cost, const struct UpdateArgs *args, struct Probe *probe);
int backwardCost(struct Cost *cost, const struct Tensor *gt, const struct UpdateArgs *args, struct Probe *probe);