#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include<sys/time.h>
#include <unistd.h>
#include <errno.h>

#include "debug_macros.h"
#include "layer.h"
#include "cost.h"
#include "tensor.h"
#include "network.h"
#include "opt_alg.h"

struct Network {
    int n_layers;
    struct Layer **layers; // Network不负责释放这部分内存
    struct Cost *cost;  // Network不负责释放这部分内存
    struct Tensor **outputs;
    struct Tensor **deltas;
    struct UpdateArgs update_args;
};

int createNetwork(struct Network **network, struct Layer **layers, int n_layers, struct Cost *cost, const struct UpdateArgs *args)
{
    CHK_NIL(layers);
    CHK_ERR((n_layers > 0)? 0: 1);
    CHK_NIL(cost);
    CHK_ERR(checkUpdateArgs(args));

    // 检查相邻层神经元个数是否匹配
    int i = 0;
    for (i = 0; i < n_layers - 1; ++i) {
        int n_bottom = 0;
        int n_top = 0;
        CHK_ERR(getLayerOutputNumber(&n_bottom, layers[i]));
        CHK_ERR(getLayerInputNumber(&n_top, layers[i + 1]));
        if (n_bottom != n_top) {
            ERR_MSG("layer[%d] n_out = %d, layer[%d] n_input = %d, size not match, error.\n", i, n_bottom, i + 1, n_top);
            return ERR_COD;
        }
    }

    // 创建Network对象
    struct Network *net = calloc(1, sizeof(struct Network));
    if (!net) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    net->layers = layers;
    net->n_layers = n_layers;
    net->cost = cost;
    (net->update_args).batch_size = args->batch_size;
    (net->update_args).lr = (args->lr) / args->batch_size; // 学习速率依batch递减, 模仿yolov2
    (net->update_args).momentum = args->momentum;

    // 创建其他部分
    CHK_NIL_GOTO((net->layers = calloc(n_layers, sizeof(struct Layer *))));
    for (i = 0; i < n_layers; ++i) {
        net->layers[i] = layers[i];
    }

    CHK_NIL_GOTO((net->deltas = calloc(n_layers, sizeof(struct Tensor *))));
    CHK_NIL_GOTO((net->outputs = calloc(n_layers, sizeof(struct Tensor *))));

    for (i = 0; i < n_layers; ++i) {
        int n_out = 0;
        CHK_ERR_GOTO(getLayerOutputNumber(&n_out, net->layers[i]));
        CHK_ERR_GOTO(createTensor(&(net->deltas[i]), args->batch_size, n_out, 1, 1));
        CHK_ERR_GOTO(createTensor(&(net->outputs[i]), args->batch_size, n_out, 1, 1));
    }

    // 连接各层, 实质上就是将各层与其正向传播和反向传播时的输入输出Tensor关联
    // 注意: 输入层layers[0]没有input和delta_ou
    for (i = 0; i < n_layers - 1; ++i) {
        CHK_ERR_GOTO(setLayerOutput(net->layers[i], net->outputs[i]));
        CHK_ERR_GOTO(setLayerInputDelta(net->layers[i], net->deltas[i]));
    }
    for (i = 1; i < n_layers - 1; ++i) {
        CHK_ERR_GOTO(setLayerInput(net->layers[i], net->outputs[i - 1]));
        CHK_ERR_GOTO(setLayerOutputDelta(net->layers[i], net->deltas[i - 1]));
    }
    CHK_ERR_GOTO(setCostInput(net->cost, net->outputs[n_layers - 1]));
    CHK_ERR_GOTO(setCostDelta(net->cost, net->deltas[n_layers - 1]));

    *network = net;
    return SUCCESS;

err_end:
    if (net) {
        if (net->outputs) {
            for (i = 0; i < n_layers - 1; ++i) {
                destroyTensor(net->outputs[i]);
            }
        }
        if (net->deltas) {
            for (i = 0; i < n_layers - 1; ++i) {
                destroyTensor(net->deltas[i]);
            }
        }
        free(net->outputs);
        free(net->deltas);
        free(net->layers);
    }
    free(net);
    return ERR_COD;
}

void destroyNetwork(struct Network *net)
{
    if (net) {
        int i;
        if (net->outputs) {
            for (i = 0; i < net->n_layers; ++i) {
                destroyTensor(net->outputs[i]);
            }
        }
        free(net->outputs);
        if (net->deltas) {
            for (i = 0; i < net->n_layers; ++i) {
                destroyTensor(net->deltas[i]);
            }
        }
        free(net->deltas);
        free(net->layers);
    }
    free(net);
}
 
int forwardNetwork(struct Network *net, const struct Tensor *input)
{
    CHK_NIL(net);
    CHK_NIL(input);

    CHK_ERR(setLayerInput(net->layers[0], input));
    int i = 0;
    for (i = 0; i < net->n_layers; ++i) {
        CHK_ERR(forwardLayer(net->layers[i]));
    }
    CHK_ERR(forwardCost(net->cost));

    return 0;
}

int backwardNetwork(struct Network *net, const struct Tensor *gt)
{
    CHK_NIL(net);
    CHK_NIL(gt);

    CHK_ERR(backwardCost(net->cost, gt));
    int i = 0;
    for (i = net->n_layers - 1; i >=0; --i) {
        CHK_ERR(backwardLayer(net->layers[i]));
    }
    return 0;
}

int updateNetwork(struct Network *net)
{
    CHK_NIL(net);

    int i = 0;
    for (i = net->n_layers - 1; i >= 0; --i) {
        CHK_ERR(updateLayer(net->layers[i], &(net->update_args)));
    }
    return SUCCESS;
}