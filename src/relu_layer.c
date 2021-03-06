#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "math_utils.h"
#include "tensor.h"
#include "activations.h"
#include "layer.h"
#include "relu_layer.h"
#include "opt_alg.h"
#include "probe.h"
#include "const.h"


struct ReluLayer
{
    struct Layer layer;
    int n_neurons;
};

int createReluLayer(struct ReluLayer **l, const char *name)
{
    CHK_NIL(l);

    struct ReluLayer *layer = calloc(1, sizeof(struct ReluLayer));
    if (layer == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    ((struct Layer *)layer)->type = RELU_LAYER_TYPE;

    if (name) {
        snprintf(((struct Layer *)layer)->name, NN_LAYER_NAME_LEN, "%s", name);
    }

    *l = layer;
    return SUCCESS;
}

void destroyReluLayer(struct ReluLayer *layer)
{
    free(layer);
}

int getReluLayerShape(int *n_in, int *n_out, const struct ReluLayer *layer)
{
    CHK_NIL(n_in);
    CHK_NIL(n_out);
    CHK_NIL(layer);

    *n_in = layer->n_neurons;
    *n_out = layer->n_neurons;
    return SUCCESS;
}

int getReluLayerInputNumber(int *n_in, const struct ReluLayer *layer)
{
    CHK_NIL(n_in);
    CHK_NIL(layer);
    *n_in = layer->n_neurons;
    return SUCCESS;
}

int getReluLayerOutputNumber(int *n_out, const struct ReluLayer *layer)
{
    CHK_NIL(n_out);
    CHK_NIL(layer);
    *n_out = layer->n_neurons;
    return SUCCESS;
}

int setReluLayerNeuronNumber(struct ReluLayer *layer, int n_neurons)
{
    CHK_NIL(layer);
    CHK_ERR((n_neurons > 0)? 0: 1);
    layer->n_neurons = n_neurons;
    return SUCCESS;
}

/**
 * @brief 正向传播, 任务包括：(1)计算当前层线性变换后输出hidden, (2)计算当前层非线性变换后输出output
 */
int forwardReluLayer(struct ReluLayer *layer, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(layer);

    CHK_ERR(activateTensor(((struct Layer *)layer)->output, ((struct Layer *)layer)->input, RELU));

    return SUCCESS;
}

/**
 * @brief 反向传播, 任务包括：(1)计算当前层梯度gradient（保存在layer->w_grad和layer->b_grad）, (2)计算当前层灵敏度输出delta_new
 */
int backwardReluLayer(struct ReluLayer *layer, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(layer);

    // backward propagation
    CHK_ERR(deactivateTensor(((struct Layer *)layer)->delta_out, ((struct Layer *)layer)->delta_in, ((struct Layer *)layer)->input, RELU)); // 更新delta

    return SUCCESS;
}
