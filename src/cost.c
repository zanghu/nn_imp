#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "tensor.h"
#include "cost.h"
#include "ce_cost.h"
#include "opt_alg.h"
#include "probe.h"

/*
struct Cost
{
    enum CostType type;
    int n_input;

    // ref
    struct Tensor *input;
    struct Tensor *delta;
};
*/

char *getCostTypeStrFromEnum(enum CostType type)
{
    switch(type) {
        case CE_COST_TYPE:
        return "cross_entropy_cost";

        default:
        break;
    }
    return "unknow_layer";
}

int getCostInputNumber(int *n_in, const struct Cost *cost)
{
    CHK_NIL(n_in);
    CHK_NIL(cost);
    *n_in = cost->n_input;
    return SUCCESS;
}

int getCostValue(float *val, const struct Cost *cost)
{
    CHK_NIL(val);
    CHK_NIL(cost);
    *val = cost->value;
    return SUCCESS;
}

int setCostInput(struct Cost *cost, const struct Tensor *input)
{
    CHK_NIL(cost);
    CHK_NIL(input);

    cost->input = (struct Tensor *)input;
    return SUCCESS;
}

int setCostDelta(struct Cost *cost, const struct Tensor *delta)
{
    CHK_NIL(cost);
    CHK_NIL(delta);

    cost->delta = (struct Tensor *)delta;
    return SUCCESS;
}

int getCostGroundTruthAttributes(int *n_features, enum DType *dtype, const struct Cost *cost)
{
    CHK_NIL(cost);

    switch (cost->type)  {
        case CE_COST_TYPE:
        CHK_ERR(getCECostGroundTruthAttributes(n_features, dtype, (const struct CECost *)cost));
        break;

        default:
        ERR_MSG("NotImplementedError, error.\n");
        return ERR_COD;
    }
    return SUCCESS;
}

int setCostName(struct Cost *cost, const char *name)
{
    CHK_NIL(cost);
    CHK_NIL(name);

    snprintf(cost->name, NN_COST_NAME_LEN, "%s", name);
    return SUCCESS;
}

int setCostIndex(struct Cost *cost, int idx)
{
    CHK_NIL(cost);
    cost->idx = idx;
    return SUCCESS;
}

int forwardCost(struct Cost *cost, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(cost);

    switch (cost->type)  {
        case CE_COST_TYPE:
        CHK_ERR(forwardCECost((struct CECost *)cost, args, probe));
        break;

        default:
        ERR_MSG("NotImplementedError, error.\n");
        return ERR_COD;
    }
    return SUCCESS;
}

int backwardCost(struct Cost *cost, const struct Tensor *gt, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(cost);

    switch (cost->type)  {
        case CE_COST_TYPE:
        CHK_ERR(backwardCECost((struct CECost *)cost, gt, args, probe));
        break;

        default:
        ERR_MSG("NotImplementedError, error.\n");
        return ERR_COD;
    }
    return SUCCESS;
}


