#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "math_utils.h"
#include "matrix.h"
#include "activations.h"
#include "fc_layer.h"


struct FCLayer {
    struct Matrix *w; // n_inputs * n_outputs，布局与yolov2保持一致
    struct Matrix *b;
    struct Matrix *w_grad; // n_inputs * n_outputs，布局与yolov2保持一致
    struct Matrix *b_grad;
    enum ActivationType act_type;

    // ref only, memory not own 
    struct Matrix *input;
    struct Matrix *output;
};

struct FCLayer *createFCLayer(int n_in, int n_out, const char *act_str)
{
    if (n_in <= 0) return NULL;
    if (n_in <= 0) return NULL;
    if (act_str == NULL) return NULL;

    struct FCLayer *layer = calloc(1, sizeof(struct FCLayer));
    if (layer == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return NULL;
    }
    layer->act_type = getActivationEnumFromStr(act_str);
    if (layer->act_type == ACT_UNKNOW) {
        ERR_MSG("Unknow activation type: %s found, error.\n", act_str);
        goto err_end;
    }

    // Weight
    layer->w = createMatrix(1, n_in, n_out, 1); // n_inputs * n_outputs，布局与yolov2保持一致
    if (layer->w == NULL) {
        ERR_MSG("createMatrix() failed, error.\n");
        goto err_end;
    }
    initMatrixParameterAsWeight(layer->w);

    // Bias
    layer->b = createMatrix(1, 1, n_out, 1);
    if (layer->w == NULL) {
        ERR_MSG("createMatrix() failed, error.\n");
        goto err_end;
    }
    initMatrixParameterAsBias(layer->b);

    // Weight Gradient
    layer->w_grad = createMatrix(1, n_in, n_out, 1); // n_inputs * n_outputs，布局与yolov2保持一致
    if (layer->w_grad == NULL) {
        ERR_MSG("createMatrix() failed, error.\n");
        goto err_end;
    }

    // Bias Gradient
    layer->b_grad = createMatrix(1, 1, n_out, 1);
    if (layer->b_grad == NULL) {
        ERR_MSG("createMatrix() failed, error.\n");
        goto err_end;
    }

    return layer;

err_end:
    if (layer) {
        free(layer->b_grad);
        free(layer->w_grad);
        free(layer->b);
        free(layer->w);
    }
    free(layer);
    return NULL;
}

void destroyFCLayer(struct FCLayer *layer)
{
    if (layer) {
        free(layer->b_grad);
        free(layer->w_grad);
        free(layer->b);
        free(layer->w);
    }
    free(layer);
}

int getFCLayerInputNeuronNumber(const struct FCLayer *layer)
{
    CHK_NIL(layer);
    return getMatrixCol(layer->w);
}

int getFCLayerOutputNeuronNumber(const struct FCLayer *layer)
{
    CHK_NIL(layer);
    return getMatrixRow(layer->w);
}

/**
 * @brief 正向传播, 任务包括：(1)计算当前层线性变换后输出hidden, (2)计算当前层非线性变换后输出output
 */
int forwardFCLayer(struct Matrix *output, struct Matrix *hidden, struct FCLayer *layer, const struct Matrix *input)
{
    CHK_NIL(output);
    CHK_NIL(hidden);
    CHK_NIL(layer);
    CHK_NIL(input);

    CHK_ERR(linearMatrix(hidden, layer->w, input, layer->b));
    CHK_ERR(activateMatrix(output, hidden, layer->act_type));

    layer->input = (struct Matrix *)input; // add input ref
    layer->output = (struct Matrix *)output; // add output ref

    return SUCCESS;
}

/**
 * @brief 反向传播, 任务包括：(1)计算当前层梯度gradient（保存在layer->w_grad和layer->b_grad）, (2)计算当前层灵敏度输出delta_new
 */
int backwardFCLayer(struct Matrix *delta_new, const struct Matrix *output, struct FCLayer *layer, const struct Matrix *delta)
{
    CHK_NIL(delta_new);
    CHK_NIL(output);
    CHK_NIL(layer);
    CHK_NIL(delta);

    // backward propagation
    CHK_ERR(deactivateMatrix(delta, output, layer->act_type)); // 更新delta, 输入的delta变为乘以反向非线性变换导数向量后的delta
    CHK_ERR(linearMatrix(delta_new, layer->w, delta, layer->b));

    // update gradient
    CHK_ERR(mulMatrix(layer->w_grad, layer->input, delta)); // update weight gradient
    CHK_ERR(mulMatrixPointwiseAndSum(layer->b_grad, delta, layer->input)); // update bias gradient

    return SUCCESS;
}

int updateFCLayer(struct FCLayer *layer, float lr, float momentum)
{
    CHK_NIL(layer);
    // 注意：这里的lr应该是已经除以了batch_size后的lr
    CHK_ERR(addMatrix(layer->w, layer->w_grad, lr, momentum));
    CHK_ERR(addMatrix(layer->b, layer->b_grad, lr, momentum));
    return SUCCESS;
}
