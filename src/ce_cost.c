#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "tensor.h"
#include "cost.h"
#include "ce_cost.h"
#include "probe.h"
#include "opt_alg.h"

struct CECost
{
    struct Cost base;
    struct Tensor *p; // batch个样本的分类概率向量组成的矩阵, 在首次运行时动态创建，不会重复创建
};

int createCECost(struct CECost **c, int n_classes)
{
    CHK_NIL(c);
    CHK_ERR((n_classes > 0)? 0: 1);

    struct CECost *cost = calloc(1, sizeof(struct CECost));
    if (cost == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    ((struct Cost *)cost)->type = CE_COST_TYPE;
    ((struct Cost *)cost)->n_input = n_classes;

    *c = cost;
    return SUCCESS;
}

void destroyCECost(struct CECost *cost)
{
    if (cost) {
        destroyTensor(cost->p);
    }
    free(cost);
}

int getCECostGroundTruthAttributes(int *n_features, enum DType *dtype, const struct CECost *cost)
{
    CHK_NIL(n_features);
    CHK_NIL(dtype);
    CHK_NIL(cost);

    *n_features = ((const struct Cost *)cost)->n_input;
    *dtype = UINT8;

    return SUCCESS;
}

int forwardCECost(struct CECost *cost, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(cost);
    if (cost->p == NULL) {
        int b;
        CHK_ERR(getTensorBatch(&b, ((struct Cost *)cost)->input));
        CHK_ERR(createTensor(&(cost->p), b, 1, ((struct Cost *)cost)->n_input, 1));
    }
    CHK_ERR(softmaxTensor(cost->p, ((struct Cost *)cost)->input)); // 计算概率向量y
    if (probe->sw_p_class) {
        CHK_ERR(copyTensorData(probe->p_class, FLOAT32, cost->p));
    }
    return SUCCESS;
}

int backwardCECost(struct CECost *cost, const struct Tensor *gt, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(cost);
    CHK_NIL(gt);

    CHK_ERR(addTensor2(((struct Cost *)cost)->delta, cost->p, gt)); // 计算反向传播的初始灵敏度delta
    CHK_ERR(probTensor(&(((struct Cost *)cost)->value), cost->p, gt)); // 计算代价值: batch的对数似然
    if (probe->sw_ce_cost) {
        probe->ce_cost = ((struct Cost *)cost)->value;
    }

    return SUCCESS;
}

