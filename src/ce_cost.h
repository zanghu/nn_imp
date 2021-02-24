#pragma once

#include "tensor.h"
#include "probe.h"
#include "opt_alg.h"

struct CECost;

int createCECost(struct CECost **c, const char *name, int n_classes);
void destroyCECost(struct CECost *cost);
int getCECostGroundTruthAttributes(int *n_features, enum DType *dtype, const struct CECost *cost);
int getCECostClassProbabilityConstRef(const float *(*p), const struct CECost *cost);

int forwardCECost(struct CECost *cost, const struct UpdateArgs *args, struct Probe *probe);
int backwardCECost(struct CECost *cost, const struct Tensor *gt, const struct UpdateArgs *args, struct Probe *probe);