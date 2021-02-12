#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include<sys/time.h>
#include <unistd.h>
#include <errno.h>

#include "layer.h"
#include "matrix.h"
#include "network.h"
#include "debug_macros.h"


struct Network {
    int n_layers;
    struct Layer **layers;
    struct Matrix **hiddens;
    struct Matrix **outputs;
    struct Matrix **deltas;
};

struct Network *createNetwork(struct Layer **layers, int n_layers, int batch_size)
{
    if (!layers) return NULL;
    if (n_layers <= 0) return NULL;
    if (batch_size <= 0) return NULL;

    // 检查相邻层神经元个数是否匹配
    int i = 0;
    for (i = 0; i < n_layers - 1; ++i) {
        int n_bottom = getLayerOutputNeuronNumber(layers[i]);
        int n_top = getLayerInputNeuronNumber(layers[i + 1]);
        if (n_bottom != n_top) {
            ERR_MSG("layer[%d] n_out = %d, layer[%d] n_input = %d, size not match, error.\n", i, n_bottom, i + 1, n_top);
            return NULL;
        }
    }

    // 创建Network对象
    struct Network *net = calloc(1, sizeof(struct Network));
    if (!net) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return NULL;
    }
    net->layers = layers;
    net->n_layers = n_layers;

    // 创建其他部分
    net->deltas = calloc(n_layers, sizeof(struct Matrix *));
    if (net->deltas == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        goto err_end;
    }

    net->hiddens = calloc(n_layers, sizeof(struct Matrix *));
    if (net->hiddens == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        goto err_end;
    }

    net->outputs = calloc(n_layers, sizeof(struct Matrix *));\
    if (net->outputs == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        goto err_end;
    }

    for (i = 0; i < n_layers - 1; ++i) {
        int n_bottom = getLayerOutputNeuronNumber(net->layers[i]);
        int n_top = getLayerInputNeuronNumber(net->layers[i + 1]);
        net->deltas[i] = createMatrix(1, n_top, n_bottom, 1);
        if (net->deltas[i] == NULL) {
            ERR_MSG("createMatrix() failed, error.\n");
            goto err_end;
        }
        net->hiddens[i] = createMatrix(1, n_top, n_bottom, 1);
        if (net->hiddens[i] == NULL) {
            ERR_MSG("createMatrix() failed, error.\n");
            goto err_end;
        }
        net->outputs[i] = createMatrix(1, n_top, n_bottom, 1);
        if (net->outputs[i] == NULL) {
            ERR_MSG("createMatrix() failed, error.\n");
            goto err_end;
        }
    }

err_end:
    if (net) {
        if (net->outputs) {
            for (i = 0; i < n_layers - 1; ++i) {
                destroyMatrix(net->outputs[i]);
            }
        }
        if (net->hiddens) {
            for (i = 0; i < n_layers - 1; ++i) {
                destroyMatrix(net->hiddens[i]);
            }
        }
        if (net->deltas) {
            for (i = 0; i < n_layers - 1; ++i) {
                destroyMatrix(net->deltas[i]);
            }
        }
        free(net->outputs);
        free(net->hiddens);
        free(net->deltas);
    }
    free(net);
    return NULL;
}

static int clearNetworkFromTrain(struct Network *net)
{
    CHK_NIL(net);

    int i = 0;
    if (net->outputs) {
        for (i = 0; i < net->n_layers - 1; ++i) {
            destroyMatrix(net->outputs[i]);
        }
    }
    if (net->hiddens) {
        for (i = 0; i < net->n_layers - 1; ++i) {
            destroyMatrix(net->hiddens[i]);
        }
    }
    if (net->deltas) {
        for (i = 0; i < net->n_layers - 1; ++i) {
            destroyMatrix(net->deltas[i]);
        }
    }
    free(net->outputs);
    free(net->hiddens);
    free(net->deltas);
    return SUCCESS;
}

int resetNetworkToTrain(struct Network *net, int batch_size)
{
    CHK_NIL(net);
    CHK_ERR((batch_size > 0)? 0: 1);

    if (net->layers == NULL) {
        ERR_MSG("net->layers is NULL, error.\n");
        return ERR_COD;
    }

    CHK_ERR(clearNetworkFromTrain(net));

    net->deltas = calloc(net->n_layers, sizeof(struct Matrix *));
    if (net->deltas != NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        goto err_end;
    }
    net->hiddens = calloc(net->n_layers, sizeof(struct Matrix *));
    if (net->hiddens != NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        goto err_end;
    }
    net->outputs = calloc(net->n_layers, sizeof(struct Matrix *));
    if (net->outputs != NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        goto err_end;
    }

    int i = 0;
    for (i = 0; i < net->n_layers - 1; ++i) {
        int n_bottom = getLayerOutputNeuronNumber(net->layers[i]);
        int n_top = getLayerInputNeuronNumber(net->layers[i + 1]);
        net->deltas[i] = createMatrix(1, n_top, n_bottom, 1);
        if (net->deltas[i] == NULL) {
            ERR_MSG("createMatrix() failed, error.\n");
            goto err_end;
        }
        net->hiddens[i] = createMatrix(1, n_top, n_bottom, 1);
        if (net->hiddens[i] == NULL) {
            ERR_MSG("createMatrix() failed, error.\n");
            goto err_end;
        }
        net->outputs[i] = createMatrix(1, n_top, n_bottom, 1);
        if (net->outputs[i] == NULL) {
            ERR_MSG("createMatrix() failed, error.\n");
            goto err_end;
        }
    }

err_end:
    if (net->outputs) {
        for (i = 0; i < net->n_layers - 1; ++i) {
            destroyMatrix(net->outputs[i]);
        }
    }
    if (net->hiddens) {
        for (i = 0; i < net->n_layers - 1; ++i) {
            destroyMatrix(net->hiddens[i]);
        }
    }
    if (net->deltas) {
        for (i = 0; i < net->n_layers - 1; ++i) {
            destroyMatrix(net->deltas[i]);
        }
    }
    free(net->outputs);
    free(net->hiddens);
    free(net->deltas);
    return ERR_COD;
}
 
int forwardNetwork(struct Network *net, const struct Matrix *input)
{
    CHK_NIL(net);
    CHK_NIL(input);

    const struct Matrix *input_tmp = input;
    int i = 0;
    while (i < net->n_layers) {
        forwardLayer(net->outputs[i], net->hiddens[i], net->layers[i], input_tmp);
        input_tmp = net->outputs[i];
        ++i;
    }
    return 0;
}

int backwardNetwork(struct Network *net, const struct Matrix *delta)
{
    CHK_NIL(net);
    CHK_NIL(delta);

    const struct Matrix *delta_tmp = delta;
    int i = net->n_layers - 1;
    while (i >= 0) {
        backwardLayer(net->deltas[i], net->outputs[i], net->layers[i], delta_tmp);
        delta_tmp = net->deltas[i];
        --i;
    }
    return 0;
}