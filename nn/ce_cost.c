#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "tensor.h"
#include "cost.h"
#include "ce_cost.h"

struct CECost
{
    struct Cost base;
    struct Tensor *p; // batch个样本的分类概率向量组成的矩阵
    //struct Tensor *y_onehot; //根据p进行分类的结果，每一行是一个one-hot向量
    //struct Tensor *y;
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
    //if (cost) {
    //    destroyTensor(cost->y);
    //}
    free(cost);
}

int forwardCECost(struct CECost *cost)
{
    CHK_NIL(cost);
    CHK_ERR(softmaxTensor(cost->p, ((struct Cost *)cost)->input)); // 计算概率向量y
    return SUCCESS;
}

int backwardCECost(struct CECost *cost, const struct Tensor *gt)
{
    CHK_NIL(cost);
    CHK_NIL(gt);

    CHK_ERR(addTensor2(((struct Cost *)cost)->delta, cost->p, gt)); // 计算反向传播的初始灵敏度delta
    CHK_ERR(probTensor(&(((struct Cost *)cost)->value), cost->p, gt)); // 计算代价值: batch的对数似然

    return SUCCESS;
}

