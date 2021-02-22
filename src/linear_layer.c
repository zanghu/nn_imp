#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "math_utils.h"
#include "tensor.h"
#include "layer.h"
#include "linear_layer.h"
#include "opt_alg.h"
#include "probe.h"
#include "const.h"


struct LinearLayer
{
    // 基类，接口类
    struct Layer base;

    struct Tensor *w; // n_inputs * n_outputs，布局与yolov2保持一致
    struct Tensor *b;
    struct Tensor *w_grad; // n_inputs * n_outputs，布局与yolov2保持一致
    struct Tensor *b_grad;;
};

int createLinearLayer(struct LinearLayer **l, const char *name, int n_in, int n_out)
{
    CHK_NIL(l);
    CHK_ERR((n_in > 0)? 0: 1);
    CHK_ERR((n_out > 0)? 0: 1);

    struct LinearLayer *layer = calloc(1, sizeof(struct LinearLayer));
    if (layer == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    ((struct Layer *)layer)->type = LINEAR_LAYER_TYPE;
    if (name) {
        snprintf(((struct Layer *)layer)->name, NN_LAYER_NAME_LEN, "%s", name);
    }

    // Weight
    CHK_ERR_GOTO(createTensorParam(&(layer->w), FLOAT32, n_out, n_in)); // outputs * inputs，布局与yolov2保持一致
    CHK_ERR(initTensorParameterAsWeight(layer->w));

    // Bias
    CHK_ERR_GOTO(createTensorParam(&(layer->b), FLOAT32, 1, n_out));
    CHK_ERR(initTensorParameterAsBias(layer->b));

    // Weight Gradient
    CHK_ERR_GOTO(createTensorParam(&(layer->w_grad), FLOAT32, n_out, n_in)); // outputs * inputs，布局与yolov2保持一致

    // Bias Gradient
    CHK_ERR_GOTO(createTensorParam(&(layer->b_grad), FLOAT32, 1, n_out));

    *l = layer;
    return SUCCESS;

err_end:
    if (layer) {
        destroyTensor(layer->b_grad);
        destroyTensor(layer->w_grad);
        destroyTensor(layer->b);
        destroyTensor(layer->w);
    }
    free(layer);
    return ERR_COD;
}

void destroyLinearLayer(struct LinearLayer *layer)
{
    if (layer) {
        destroyTensor(layer->b_grad);
        destroyTensor(layer->w_grad);
        destroyTensor(layer->b);
        destroyTensor(layer->w);
    }
    free(layer);
}

int getLinearLayerShape(int *n_in, int *n_out, const struct LinearLayer *layer)
{
    CHK_NIL(n_in);
    CHK_NIL(n_out);
    CHK_NIL(layer);

    CHK_ERR(getTensorRowAndCol(n_out, n_in, layer->w));
    return SUCCESS;
}

int getLinearLayerInputNumber(int *n_in, const struct LinearLayer *layer)
{
    CHK_NIL(layer);
    CHK_ERR(getTensorCol(n_in, layer->w));
    return SUCCESS;
}

int getLinearLayerOutputNumber(int *n_out, const struct LinearLayer *layer)
{
    CHK_NIL(layer);
    CHK_ERR(getTensorRow(n_out, layer->w));
    return SUCCESS;
}

/**
 * @brief 正向传播, 计算当前层非线性变换后输出output, 相当于full_connected_layer的隐藏层神经元的值
 */
int forwardLinearLayer(struct LinearLayer *layer, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(layer);

    CHK_ERR(linearTensorForward(((struct Layer *)layer)->output, ((struct Layer *)layer)->input, layer->w, layer->b));

    if (probe->dump_output) {
        CHK_ERR(savetxtTensorData(((struct Layer *)layer)->output, probe->dst_dir, "output", ((struct Layer *)layer)->name, args->cur_epoch, args->cur_iter));
    }

    return SUCCESS;
}

/**
 * @brief 反向传播, 任务包括：(1)计算当前层梯度gradient（保存在layer->w_grad和layer->b_grad）, (2)计算当前层灵敏度输出delta_new
 */
int backwardLinearLayer(struct LinearLayer *layer, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(layer);

    // backward propagation
    if (((struct Layer *)layer)->delta_out) { // 反向传播到达layer[0]时，delta_out为NULL
        CHK_ERR(linearTensorBackward(((struct Layer *)layer)->delta_out, ((struct Layer *)layer)->delta_in, layer->w));

        if (probe->dump_delta) {
            CHK_ERR(savetxtTensorData(((struct Layer *)layer)->delta_out, probe->dst_dir, "delta", ((struct Layer *)layer)->name, args->cur_epoch, args->cur_iter));
        }
    }
    //if ((((struct Layer *)layer)->delta_out)) {
    //    fprintf(stdout, "%s addr(delta_out) = %p\n", ((struct Layer *)layer)->name, ((struct Layer *)layer)->delta_out);
    //}
    //if ((((struct Layer *)layer)->delta_in)) {
    //    fprintf(stdout, "%s addr(delta_in) = %p\n", ((struct Layer *)layer)->name, ((struct Layer *)layer)->delta_in);
    //}

    // update gradient
    //CHK_ERR(linearTensorWeightGradient(layer->w_grad, ((struct Layer *)layer)->delta_in, 1, ((struct Layer *)layer)->input, 0, NULL)); // update weight gradient
    CHK_ERR(linearTensorWeightGradient(layer->w_grad, ((struct Layer *)layer)->delta_in, ((struct Layer *)layer)->input)); // update weight gradient
    //CHK_ERR(savetxtTensorData(((struct Layer *)layer)->delta_in, probe->dst_dir, "delta_in", ((struct Layer *)layer)->name, args->cur_epoch, args->cur_iter));
    if (probe->dump_gw) {
        CHK_ERR(savetxtTensorParam(layer->w_grad, probe->dst_dir, "gW", ((struct Layer *)layer)->name, args->cur_epoch, args->cur_iter));
    }
    CHK_ERR(linearTensorBiasGradient(layer->b_grad, ((struct Layer *)layer)->delta_in)); // update bias gradient
    if (probe->dump_gb) {
        CHK_ERR(savetxtTensorParam(layer->b_grad, probe->dst_dir, "gb", ((struct Layer *)layer)->name, args->cur_epoch, args->cur_iter));
    }

    return SUCCESS;
}

/**
 * @brief: 更新当前层参数
 */
int updateLinearLayer(struct LinearLayer *layer, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(layer);
    CHK_NIL(args);
/*
    logTensorStr("@Before Paramter Update=========================\n");

    logTensorStr("Linear Weight:\n");
    logTensorParam(layer->w);
    logTensorStr("Linear Bias:\n");
    logTensorParam(layer->b);

    logTensorStr("Linear Weight Gradient:\n");
    logTensorParam(layer->w_grad);
    logTensorStr("Linear Bias Gradient:\n");
    logTensorParam(layer->b_grad);

    logTensorStr("\n");
*/
    // 注意：这里的lr应该是已经除以了batch_size后的lr
    CHK_ERR(addTensor(layer->w, layer->w_grad, 1. * (args->lr) / (args->batch_size), args->momentum));
    if (probe->dump_w) {
        CHK_ERR(savetxtTensorParam(layer->w, probe->dst_dir, "W", ((struct Layer *)layer)->name, args->cur_epoch, args->cur_iter));
    }
    CHK_ERR(addTensor(layer->b, layer->b_grad, 1. * (args->lr) / (args->batch_size), args->momentum));
    if (probe->dump_b) {
        CHK_ERR(savetxtTensorParam(layer->b, probe->dst_dir, "b", ((struct Layer *)layer)->name, args->cur_epoch, args->cur_iter));
    }
    return SUCCESS;
}
