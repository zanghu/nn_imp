#include "debug_macros.h"
#include "tensor.h"
#include "layer.h"
#include "linear_layer.h"
#include "sigmoid_layer.h"
#include "softmax_layer.h"
#include "opt_alg.h"

/*
struct Layer
{
    enum LayerType type;

    // ref only, memory not own 
    struct Tensor *input;
    struct Tensor *output;
    struct Tensor *delta_in;
    struct Tensor *delta_out;
};
*/

char *getLayerTypeStrFromEnum(enum LayerType type)
{
    switch(type) {
        case LINEAR_LAYER_TYPE:
        return "linear_layer";

        case SIGMOID_LAYER_TYPE:
        return "sigmoid_layer";

        case SOFTMAX_LAYER_TYPE:
        return "softmax_layer";

        default:
        break;
    }
    return "unknow_layer";
}

/*
void destroyLayer(struct Layer *layer)
{
    if (layer == NULL) return;
    switch (layer->type) {
        case LINEAR_LAYER_TYPE:
        destroyLinearLayer((struct LinearLayer **)layer, n_in, n_out);
        break;

        case SIGMOID_LAYER_TYPE:
        destroySigmoidLayer((struct SigmoidLayer **)layer, n_in, n_out);
        break;

        case SOFTMAX_LAYER_TYPE:
        destroySoftmaxLayer((struct SoftmaxLayer **)layer, n_in, n_out);
        break;

        default:
        ERR_MSG("Unkonw Layer Type found: %s, error.\n", getLayerTypeStrFromEnum(layer->type));
        break; // TODO: 此处未抛出异常
    }
}
*/

int forwardLayer(struct Layer *layer)
{
    CHK_NIL(layer);

    switch (layer->type) {
        case LINEAR_LAYER_TYPE:
        CHK_ERR(forwardLinearLayer((struct LinearLayer *)layer));
        break;

        case SIGMOID_LAYER_TYPE:
        CHK_ERR(forwardSigmoidLayer((struct SigmoidLayer *)layer));
        break;

        case SOFTMAX_LAYER_TYPE:
        CHK_ERR(forwardSoftmaxLayer((struct SoftmaxLayer *)layer));
        break;

        default:
        ERR_MSG("Unkonw Layer Type found: %s, error.\n", getLayerTypeStrFromEnum(layer->type));
        return ERR_COD;
    }
    return SUCCESS;
}

int backwardLayer(struct Layer *layer)
{
    CHK_NIL(layer);

    switch (layer->type) {
        case LINEAR_LAYER_TYPE:
        CHK_ERR(backwardLinearLayer((struct LinearLayer *)layer));
        break;

        case SIGMOID_LAYER_TYPE:
        CHK_ERR(backwardSigmoidLayer((struct SigmoidLayer *)layer));
        break;

        case SOFTMAX_LAYER_TYPE:
        CHK_ERR(backwardSoftmaxLayer((struct SoftmaxLayer *)layer));
        break;

        default:
        ERR_MSG("Unkonw Layer Type found: %s, error.\n", getLayerTypeStrFromEnum(layer->type));
        return ERR_COD;
    }
    return SUCCESS;
}

int updateLayer(struct Layer *layer, const struct UpdateArgs *args)
{
    CHK_NIL(layer);

    switch (layer->type) {
        case LINEAR_LAYER_TYPE:
        CHK_ERR(updateLinearLayer((struct LinearLayer *)layer, args));
        break;

        case SIGMOID_LAYER_TYPE: // sigmoid_layer无需参数更新，直接略过
        //ERR_MSG("NotImplementedError, SigmoidLayer has no update method, error.\n");
        //return ERR_COD;
        break;

        case SOFTMAX_LAYER_TYPE: // softmax_layer无需参数更新，直接略过
        //ERR_MSG("NotImplementedError, SoftmaxLayer has no update method, error.\n");
        break;

        default:
        ERR_MSG("Unkonw Layer Type found: %s, error.\n", getLayerTypeStrFromEnum(layer->type));
        return ERR_COD;
    }
    return SUCCESS;
}

int getLayerInputShape(int *b, int *row, int *col, int *c, const struct Layer *layer)
{
    CHK_NIL(b);
    CHK_NIL(row);
    CHK_NIL(col);
    CHK_NIL(c);
    CHK_NIL(layer);

    switch (layer->type) {
        case LINEAR_LAYER_TYPE:
        CHK_ERR(getLinearLayerInputShape(b, row, col, c, (const struct LinearLayer *)layer));
        break;

        case SIGMOID_LAYER_TYPE:
        ERR_MSG("sigmoid_layer has no input shape, error.\n");
        return ERR_COD;

        case SOFTMAX_LAYER_TYPE:
        ERR_MSG("softmax_layer has no input shape, error.\n");
        return ERR_COD;

        default:
        ERR_MSG("Unkonw Layer Type found: %s, error.\n", getLayerTypeStrFromEnum(layer->type));
        return ERR_COD;
    }
    return SUCCESS;
}

int getLayerInputNumber(int *n_in, const struct Layer *layer)
{
    CHK_NIL(n_in);
    CHK_NIL(layer);

    switch (layer->type) {
        case LINEAR_LAYER_TYPE:
        CHK_ERR(getLinearLayerInputNumber(n_in, (const struct LinearLayer *)layer));
        break;

        case SIGMOID_LAYER_TYPE:
        CHK_ERR(getSigmoidLayerInputNumber(n_in, (const struct SigmoidLayer *)layer));
        break;

        case SOFTMAX_LAYER_TYPE:
        CHK_ERR(getSoftmaxLayerInputNumber(n_in, (const struct SoftmaxLayer *)layer));
        break;

        default:
        ERR_MSG("Unkonw Layer Type found: %s, error.\n", getLayerTypeStrFromEnum(layer->type));
        return ERR_COD;
    }
    return SUCCESS;
}

int getLayerOutputNumber(int *n_out, const struct Layer *layer)
{
    CHK_NIL(n_out);
    CHK_NIL(layer);

    switch (layer->type) {
        case LINEAR_LAYER_TYPE:
        CHK_ERR(getLinearLayerOutputNumber(n_out, (struct LinearLayer *)layer));
        break;

        case SIGMOID_LAYER_TYPE:
        CHK_ERR(getSigmoidLayerOutputNumber(n_out, (struct SigmoidLayer *)layer));
        break;

        case SOFTMAX_LAYER_TYPE:
        CHK_ERR(getSoftmaxLayerOutputNumber(n_out, (struct SoftmaxLayer *)layer));
        break;

        default:
        ERR_MSG("Unkonw Layer Type found: %s, error.\n", getLayerTypeStrFromEnum(layer->type));
        return ERR_COD;
    }
    return SUCCESS;
}


int setLayerInput(struct Layer *layer, const struct Tensor *input)
{
    CHK_NIL(layer);
    CHK_NIL(input);

    layer->input = (struct Tensor *)input;
    return SUCCESS;
}
    
int setLayerOutput(struct Layer *layer, const struct Tensor *output)
{
    CHK_NIL(layer);
    CHK_NIL(output);

    layer->output = (struct Tensor *)output;
    return SUCCESS;
}

int setLayerInputDelta(struct Layer *layer, const struct Tensor *delta_in)
{
    CHK_NIL(layer);
    CHK_NIL(delta_in);

    layer->delta_in = (struct Tensor *)delta_in;
    return SUCCESS;
}

int setLayerOutputDelta(struct Layer *layer, const struct Tensor *delta_out)
{
    CHK_NIL(layer);
    CHK_NIL(delta_out);

    layer->delta_out = (struct Tensor *)delta_out;
    return SUCCESS;
}

/*
int setLayerOptAlgArgs(struct Layer *layer, float lr, float momentum)
{
    CHK_NIL(layer);

    layer->lr = lr;
    layer->momentum = momentum;

    return SUCCESS;
}
*/