#pragma once

#include "tensor.h"

struct CECost;

int createCECost(struct CECost **c, int n_classes);
void destroyCECost(struct CECost *cost);

int forwardCECost(struct CECost *cost);
int backwardCECost(struct CECost *cost, const struct Tensor *gt);