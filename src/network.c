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
#include "probe.h"

struct Network {
    int n_layers;
    struct Layer **layers; // Network不负责释放这部分内存
    struct Cost *cost;  // Network不负责释放这部分内存
    struct Tensor **outputs;
    struct Tensor **deltas;
    struct Tensor *input; // 输入数据缓存
    struct Tensor *gt; // 样本真值缓存
};

int createNetwork(struct Network **network, struct Layer **layers, int n_layers, struct Cost *cost)
{
    CHK_NIL(layers);
    CHK_ERR((n_layers > 0)? 0: 1);
    CHK_NIL(cost);

    // 检查每一层的名字是否唯一
    int i = 0;
    for (i = 0; i < n_layers; ++i) {
        const char *name_i = NULL;
        CHK_ERR(getLayerName(&name_i, layers[i]));
        int j = 0;
        for (j = i + 1; j < n_layers; ++j) {
            const char *name_j = NULL;
            CHK_ERR(getLayerName(&name_j, layers[j]));
            if (strcmp(name_i, name_j) == 0) {
                ERR_MSG("Layers[%d] name: %s is the same as Layer[%d] name: %s, error.\n", i, name_i, j, name_j);
                return ERR_COD;
            }
        }
    }

    // 根据前一层神经元个数设置激活层神经元个数
    for (i = 1; i < n_layers; ++i) {
        int n_out = 0;
        CHK_ERR(getLayerOutputNumber(&n_out, layers[i-1]));
        CHK_ERR(setLayerNeuronNumber(layers[i], n_out));
    }

    // 检查相邻层神经元个数是否匹配
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

    // 创建其他部分
    CHK_NIL_GOTO((net->layers = calloc(n_layers, sizeof(struct Layer *))));
    for (i = 0; i < n_layers; ++i) {
        net->layers[i] = layers[i];
    }

    *network = net;
    return SUCCESS;

err_end:
    if (net) {
        free(net->layers);
    }
    free(net);
    return ERR_COD;
}

static int allocNetworkCache(struct Network *net, const struct UpdateArgs *args)
{
    CHK_NIL(net);
    CHK_NIL(args);

    int n_layers = net->n_layers;
    int i;

    CHK_NIL_GOTO((net->deltas = calloc(n_layers, sizeof(struct Tensor *))));
    CHK_NIL_GOTO((net->outputs = calloc(n_layers, sizeof(struct Tensor *))));

    for (i = 0; i < n_layers; ++i) {
        int n_out = 0;
        CHK_ERR_GOTO(getLayerOutputNumber(&n_out, net->layers[i]));
        CHK_ERR_GOTO(createTensorData(&(net->deltas[i]), FLOAT32, args->batch_size, n_out));
        CHK_ERR_GOTO(createTensorData(&(net->outputs[i]), FLOAT32, args->batch_size, n_out));
    }

    // 连接各层, 实质上就是将各层与其正向传播和反向传播时的输入输出Tensor关联
    for (i = 0; i < n_layers; ++i) {
        CHK_ERR_GOTO(setLayerOutput(net->layers[i], net->outputs[i]));
        CHK_ERR_GOTO(setLayerInputDelta(net->layers[i], net->deltas[i]));
    }
    // 注意: 输入层layers[0]没有input和delta_ou
    for (i = 1; i < n_layers; ++i) {
        CHK_ERR_GOTO(setLayerInput(net->layers[i], net->outputs[i - 1]));
        CHK_ERR_GOTO(setLayerOutputDelta(net->layers[i], net->deltas[i - 1]));
    }
    CHK_ERR_GOTO(setCostInput(net->cost, net->outputs[n_layers - 1]));
    CHK_ERR_GOTO(setCostDelta(net->cost, net->deltas[n_layers - 1]));

    return SUCCESS;

err_end:
    if (net->outputs) {
        for (i = 0; i < net->n_layers - 1; ++i) {
            destroyTensor(net->outputs[i]);
        }
    }
    if (net->deltas) {
        for (i = 0; i < net->n_layers - 1; ++i) {
            destroyTensor(net->deltas[i]);
        }
    }
    free(net->outputs);
    free(net->deltas);

    return ERR_COD;
}

static void freeNetworkCache(struct Network *net)
{
    if (net) {
        if (net->outputs) {
            int i;
            for (i = 0; i < net->n_layers - 1; ++i) {
                destroyTensor(net->outputs[i]);
            }
        }
        if (net->deltas) {
            int i;
            for (i = 0; i < net->n_layers - 1; ++i) {
                destroyTensor(net->deltas[i]);
            }
        }
        free(net->outputs);
        free(net->deltas);
    }
}

static int reallocNetworkCache(struct Network *net, const struct UpdateArgs *args)
{
    CHK_NIL(net);
    CHK_NIL(args);
    CHK_NIL(net->layers);
    CHK_ERR((net->n_layers > 0)? 0: 1);

    int need_realloc = 0;
    if (net->outputs) {
        int b;
        CHK_ERR(getTensorBatch(&b, net->outputs[0]));
        if (b < args->batch_size) { // 原有的缓冲区空间不足，需要重新分配
            need_realloc =1;
        }
    }
    else {
        need_realloc = 1;
    }

    if (need_realloc) {
        fprintf(stdout, "Network need to be reallocate for some reason...\n");
        freeNetworkCache(net);
        CHK_ERR(allocNetworkCache(net, args));
    }

    return SUCCESS;
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
        free(net->input);
        free(net->gt);
    }
    free(net);
}

/*
static int checkNetworkInput(struct Network *net, const struct Tensor *input)
{
    CHK_NIL(net);
    CHK_NIL(input);

    CHK_NIL(net->layers);
    CHK_NIL(net->layers[0]);
    int b0, col0, c0;
    int b1, row1, col1, c1;
    CHK_ERR(getLayerInputNumber(&col0, net->layers[0]));
    CHK_ERR(getTensorBatchAndChannel(&b0, &c0, net->outputs[0]));
    CHK_ERR(getTensorShape(&b0, &row1, &col1, &c0, input));
    //fprintf(stdout, "layer.shape = (%d, %d, %d, %d)\n", b0, 1, col0, c0);
    CHK_ERR(getTensorShape(&b1, &row1, &col1, &c1, input));
    //fprintf(stdout, "input.shape = (%d, %d, %d, %d)\n", b1, row1, col1, c1);
    CHK_ERR((b0 == b1)? 0:1);
    CHK_ERR((col0 == col1)? 0:1);
    CHK_ERR((c0 == c1)? 0:1);

    return SUCCESS;
}
*/
 
int forwardNetwork(struct Network *net, const void *input_data, int n_samples, int n_features, const char *dtype_str, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(net);
    CHK_NIL(input_data);
    CHK_ERR((n_samples > 0)? 0: 1);
    CHK_ERR((n_features > 0)? 0: 1);
    CHK_NIL(dtype_str);
    CHK_ERR(checkUpdateArgs(args));
    CHK_NIL(net->layers);
    CHK_NIL(args);
    CHK_ERR((n_samples <= args->batch_size)? 0: 1);

    // 每次运行时，检查是否需要分配输入输出缓冲区空间，可能的原因包括：
    // (1)网络首次运行;
    // (2)batch_size发生变化，且大于之前分配的缓冲区大小。
    CHK_ERR(reallocNetworkCache(net, args));

    // 输入数据绑定Tensor对象
    enum DType dtype = getTensorDtypeEnumFromStr(dtype_str);
    if (net->input) {
        void *blob_old = NULL; // 约定了输入的数据由用户负责保管句柄，因此不需要把旧训练数据的句柄返还给用户
        CHK_ERR(setTensorSamplesByReplace(&blob_old, net->input, (void *)input_data, n_samples, n_features, dtype));
    }
    else {
        int n_in;
        CHK_ERR(getLayerInputNumber(&n_in, net->layers[0]));
        CHK_ERR(createTensorDataWithBlobRef(&(net->input), (void *)input_data, dtype, args->batch_size, n_features, n_samples));
    }

    CHK_ERR(setLayerInput(net->layers[0], net->input));
    int i = 0;
    for (i = 0; i < net->n_layers; ++i) {
        fprintf(stdout, "forward layer %d...\n", i);
        CHK_ERR(forwardLayer(net->layers[i], args, probe));
    }
    CHK_ERR(forwardCost(net->cost, args, probe));

    return 0;
}

int backwardNetwork(struct Network *net, const void *gt_data, int n_samples, int n_features, const char *dtype_str, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(net);
    CHK_NIL(gt_data);
    CHK_ERR((n_samples > 0)? 0: 1);
    CHK_ERR((n_features > 0)? 0: 1);
    CHK_ERR(checkUpdateArgs(args));
    CHK_NIL(net->layers);
    CHK_NIL(args);
    CHK_ERR((n_samples <= args->batch_size)? 0: 1);

    // gt绑定Tensor对象
    enum DType dtype = getTensorDtypeEnumFromStr(dtype_str);
    if (net->gt) {
        void *blob_old = NULL; // 约定了输入的数据由用户负责保管句柄，因此不需要把旧训练数据的句柄返还给用户
        CHK_ERR(setTensorSamplesByReplace(&blob_old, net->gt, (void *)gt_data, n_samples, n_features, dtype));
    }
    else {
        int n_features;
        enum DType dtype_needed;
        CHK_ERR(getCostGroundTruthAttributes(&n_features, &dtype_needed,net->cost));
        CHK_ERR((dtype == dtype_needed)? 0: 1);
        //CHK_ERR(createTensorWithDataRef(&(net->gt), n_samples, 1, n_features, 1, gt_data, dtype));
        CHK_ERR(createTensorDataWithBlobRef(&(net->gt), (void *)gt_data, dtype, args->batch_size, n_features, n_samples));
    }

    CHK_ERR(backwardCost(net->cost, net->gt, args, probe));
    int i = 0;
    for (i = net->n_layers - 1; i >=0; --i) {
        fprintf(stdout, "backward layer %d...\n", i);
        CHK_ERR(backwardLayer(net->layers[i], args, probe));
    }
    return 0;
}

int updateNetwork(struct Network *net, const struct UpdateArgs *args, struct Probe *probe)
{
    CHK_NIL(net);

    int i;
    for (i = net->n_layers - 1; i >= 0; --i) {
        CHK_ERR(updateLayer(net->layers[i], args, probe));
    }
    return SUCCESS;
}
