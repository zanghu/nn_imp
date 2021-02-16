#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "math_utils.h"
#include "tensor.h"
#include "activations.h"
#include "layer.h"
#include "sigmoid_layer.h"


struct SoftmaxLayer
{
    struct Layer layer;
    int n_neurons;
};

int createSoftmaxLayer(struct SoftmaxLayer **l, int n_neurons)
{
    CHK_NIL(l);
    CHK_ERR((n_neurons > 0)? 0: 1);

    struct SoftmaxLayer *layer = calloc(1, sizeof(struct SoftmaxLayer));
    if (layer == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    ((struct Layer *)layer)->type = SOFTMAX_LAYER_TYPE;
    layer->n_neurons = n_neurons;

    *l = layer;
    return SUCCESS;
}

void destroySoftmaxLayer(struct SoftmaxLayer *layer)
{
    free(layer);
}

int getSoftmaxLayerShape(int *n_in, int *n_out, const struct SoftmaxLayer *layer)
{
    CHK_NIL(n_in);
    CHK_NIL(n_out);
    CHK_NIL(layer);

    *n_in = layer->n_neurons;
    *n_out = layer->n_neurons;
    return SUCCESS;
}

int getSoftmaxLayerInputNumber(int *n_in, const struct SoftmaxLayer *layer)
{
    CHK_NIL(n_in);
    CHK_NIL(layer);

    *n_in = layer->n_neurons;
    return SUCCESS;
}

int getSoftmaxLayerOutputNumber(int *n_out, const struct SoftmaxLayer *layer)
{
    CHK_NIL(n_out);
    CHK_NIL(layer);

    *n_out = layer->n_neurons;
    return SUCCESS;
}

/**
 * @brief 正向传播, 任务包括：(1)计算当前层线性变换后输出hidden, (2)计算当前层非线性变换后输出output
 */
int forwardSoftmaxLayer(struct SoftmaxLayer *layer, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(layer);

    //CHK_ERR(activateTensor(((struct Layer *)layer)->output, ((struct Layer *)layer)->input, LOGISTIC));
    CHK_ERR(softmaxTensor(((struct Layer *)layer)->output, ((struct Layer *)layer)->input));

    return SUCCESS;
}

/**
 * @brief 反向传播, 任务包括：(1)计算当前层梯度gradient（保存在layer->w_grad和layer->b_grad）, (2)计算当前层灵敏度输出delta_new
 */
int backwardSoftmaxLayer(struct SoftmaxLayer *layer, const struct UpdateArgs *args, struct Probe *probe)
{
    ERR_MSG("NotImplementedError, error.\n");
    return ERR_COD;
}
