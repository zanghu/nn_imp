#include "debug_macros.h"
#include "tensor.h"
#include "layer.h"
#include "linear_layer.h"
#include "sigmoid_layer.h"
#include "relu_layer.h"
#include "softmax_layer.h"
#include "opt_alg.h"
#include "probe.h"

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

        case RELU_LAYER_TYPE:
        return "relu_layer";

        case SOFTMAX_LAYER_TYPE:
        return "softmax_layer";

        default:
        break;
    }
    return "unknow_layer";
}

int forwardLayer(struct Layer *layer, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(layer);

    switch (layer->type) {
        case LINEAR_LAYER_TYPE:
        CHK_ERR(forwardLinearLayer((struct LinearLayer *)layer, args, probe));
        break;

        case SIGMOID_LAYER_TYPE:
        CHK_ERR(forwardSigmoidLayer((struct SigmoidLayer *)layer, args, probe));
        break;

        case RELU_LAYER_TYPE:
        CHK_ERR(forwardReluLayer((struct ReluLayer *)layer, args, probe));
        break;

        case SOFTMAX_LAYER_TYPE:
        CHK_ERR(forwardSoftmaxLayer((struct SoftmaxLayer *)layer, args, probe));
        break;

        default:
        ERR_MSG("Unkonw Layer Type found: %s, error.\n", getLayerTypeStrFromEnum(layer->type));
        return ERR_COD;
    }
    return SUCCESS;
}

int backwardLayer(struct Layer *layer, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(layer);

    switch (layer->type) {
        case LINEAR_LAYER_TYPE:
        CHK_ERR(backwardLinearLayer((struct LinearLayer *)layer, args, probe));
        break;

        case SIGMOID_LAYER_TYPE:
        CHK_ERR(backwardSigmoidLayer((struct SigmoidLayer *)layer, args, probe));
        break;

        case RELU_LAYER_TYPE:
        CHK_ERR(backwardReluLayer((struct ReluLayer *)layer, args, probe));
        break;

        case SOFTMAX_LAYER_TYPE:
        CHK_ERR(backwardSoftmaxLayer((struct SoftmaxLayer *)layer, args, probe));
        break;

        default:
        ERR_MSG("Unkonw Layer Type found: %s, error.\n", getLayerTypeStrFromEnum(layer->type));
        return ERR_COD;
    }
    return SUCCESS;
}

int updateLayer(struct Layer *layer, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(layer);

    switch (layer->type) {
        case LINEAR_LAYER_TYPE:
        CHK_ERR(updateLinearLayer((struct LinearLayer *)layer, args, probe));
        break;

        case SIGMOID_LAYER_TYPE: // sigmoid_layer无需参数更新，直接略过
        break;

        case RELU_LAYER_TYPE: // relu_layer无需参数更新，直接略过
        break;

        case SOFTMAX_LAYER_TYPE: // softmax_layer无需参数更新，直接略过
        break;

        default:
        ERR_MSG("Unkonw Layer Type found: %s, error.\n", getLayerTypeStrFromEnum(layer->type));
        return ERR_COD;
    }
    return SUCCESS;
}

int getLayerShape(int *n_in, int *n_out, const struct Layer *layer)
{
    CHK_NIL(n_in);
    CHK_NIL(n_out);
    CHK_NIL(layer);

    switch (layer->type) {
        case LINEAR_LAYER_TYPE:
        CHK_ERR(getLinearLayerShape(n_in, n_out, (const struct LinearLayer *)layer));
        break;

        case SIGMOID_LAYER_TYPE:
        CHK_ERR(getSigmoidLayerShape(n_in, n_out, (const struct SigmoidLayer *)layer));
        break;

        case RELU_LAYER_TYPE:
        CHK_ERR(getReluLayerShape(n_in, n_out, (const struct ReluLayer *)layer));
        break;

        case SOFTMAX_LAYER_TYPE:
        CHK_ERR(getSoftmaxLayerShape(n_in, n_out, (const struct SoftmaxLayer *)layer));
        break;

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

        case RELU_LAYER_TYPE:
        CHK_ERR(getReluLayerInputNumber(n_in, (const struct ReluLayer *)layer));
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

        case RELU_LAYER_TYPE:
        CHK_ERR(getReluLayerOutputNumber(n_out, (struct ReluLayer *)layer));
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

int getLayerName(const char *(*name), const struct Layer *layer)
{
    CHK_NIL(name);
    CHK_NIL(layer);
    *name = layer->name;
    return SUCCESS;
}

int setLayerNeuronNumber(struct Layer *layer, int n_neurons)
{
    CHK_NIL(layer);

    switch (layer->type) {
        case LINEAR_LAYER_TYPE: // 线性层不需要依赖前一层的神经元设置自己的神经元个数
        break;

        case SIGMOID_LAYER_TYPE:
        CHK_ERR(setSigmoidLayerNeuronNumber((struct SigmoidLayer *)layer, n_neurons));
        break;

        case RELU_LAYER_TYPE:
        CHK_ERR(setReluLayerNeuronNumber((struct ReluLayer *)layer, n_neurons));
        break;

        case SOFTMAX_LAYER_TYPE:
        CHK_ERR(setSoftmaxLayerNeuronNumber((struct SoftmaxLayer *)layer, n_neurons));
        break;

        default:
        ERR_MSG("Unkonw Layer Type found: %s, error.\n", getLayerTypeStrFromEnum(layer->type));
        return ERR_COD;
    }
    return SUCCESS;
}

int setLayerName(struct Layer *layer, const char *name)
{
    CHK_NIL(layer);
    CHK_NIL(name);

    snprintf(layer->name, NN_LAYER_NAME_LEN, "%s", name);

    return SUCCESS;
}

int setLayerIndex(struct Layer *layer, int idx)
{
    CHK_NIL(layer);
    layer->idx = idx;
    return SUCCESS;
}

int setLayerInput(struct Layer *layer, const struct Tensor *input)
{
    CHK_NIL(layer);
    CHK_NIL(input);
    enum TensorType type;
    CHK_ERR(getTensorType(&type, input));
    CHK_ERR((type == DATA_TENSOR_TYPE)? 0: 1);
    layer->input = (struct Tensor *)input;
    return SUCCESS;
}
    
int setLayerOutput(struct Layer *layer, const struct Tensor *output)
{
    CHK_NIL(layer);
    CHK_NIL(output);
    enum TensorType type;
    CHK_ERR(getTensorType(&type, output));
    CHK_ERR((type == DATA_TENSOR_TYPE)? 0: 1);
    layer->output = (struct Tensor *)output;
    return SUCCESS;
}

int setLayerInputDelta(struct Layer *layer, const struct Tensor *delta_in)
{
    CHK_NIL(layer);
    CHK_NIL(delta_in);
    enum TensorType type;
    CHK_ERR(getTensorType(&type, delta_in));
    CHK_ERR((type == DATA_TENSOR_TYPE)? 0: 1);
    layer->delta_in = (struct Tensor *)delta_in;
    return SUCCESS;
}

int setLayerOutputDelta(struct Layer *layer, const struct Tensor *delta_out)
{
    CHK_NIL(layer);
    CHK_NIL(delta_out);
    enum TensorType type;
    CHK_ERR(getTensorType(&type, delta_out));
    CHK_ERR((type == DATA_TENSOR_TYPE)? 0: 1);
    layer->delta_out = (struct Tensor *)delta_out;
    return SUCCESS;
}
