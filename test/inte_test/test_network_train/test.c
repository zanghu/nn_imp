#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "network.h"
#include "layer.h"
#include "linear_layer.h"
#include "sigmoid_layer.h"
#include "cost.h"
#include "ce_cost.h"
#include "opt_alg.h"
#include "probe.h"
#include "mnist.h"
#include "data_utils.h"
#include "debug_macros.h"

#define DATASET_DIR ("/home/zanghu/data_base/mnist")

int main()
{
    // 创建组件
    struct LinearLayer *linear_0 = NULL;
    CHK_ERR(createLinearLayer(&linear_0, 28 * 28, 625));
    struct SigmoidLayer *sigmoid_0 = NULL;
    CHK_ERR(createSigmoidLayer(&sigmoid_0, 625));
    struct LinearLayer *linear_1 = NULL;
    CHK_ERR(createLinearLayer(&linear_1, 625, 10));
    struct CECost *ce_cost = NULL;
    CHK_ERR(createCECost(&ce_cost, 10));
    printf("layers create finish.\n");

    // 创建网络
    struct Layer *layers[] = {(struct Layer *)linear_0, (struct Layer *)sigmoid_0, (struct Layer *)linear_1};
    struct UpdateArgs args;
    args.batch_size = 128;
    args.lr = 0.05;
    args.momentum = 0.0; // 不使用动量法
    args.n_epochs = 1000; // 最大循环数
    struct Network *net = NULL;
    CHK_ERR(createNetwork(&net, layers, 3, (struct Cost *)ce_cost));
    printf("network create finish.\n");

    // 读取数据集
    struct MNIST dataset;
    CHK_ERR(loadMnist(&dataset, DATASET_DIR));

    // 训练
    int n_train = 50000;
    //int n_valid = 10000;
    float *input = NULL;
    //unsigned char *gt = NULL;
    CHK_NIL((input = calloc(args.batch_size * MNIST_WIDTH * MNIST_HEIGHT * 1, sizeof(float))));
    //CHK_NIL((gt = calloc(args.batch_size * MNIST_N_CLASSES * 1, sizeof(unsigned char))));

    // 探针
    struct Probe probe;
    memset(&probe, 0, sizeof(struct Probe));
    probe.sw_p_class = 1;
    CHK_NIL((probe.p_class = calloc(args.batch_size * MNIST_N_CLASSES, sizeof(float))));
    probe.sw_ce_cost = 1;

    struct timeval t_train_0, t_train_1, t_train_2;
    CHK_ERR(gettimeofday(&t_train_0, NULL));
    int n_epochs = 2;
    for (int k = 0; k < n_epochs; ++k) {
        int n_iters = 0;
        const void *data = NULL;
        const void *label_onehot = NULL;
        int n_samples = 0;
        struct timeval t_epoch_0, t_epoch_1, t_epoch_2;
        CHK_ERR(gettimeofday(&t_epoch_0, NULL));
        while (1) {
            // 获取batch数据
            CHK_ERR(getMnistNthBatch((const unsigned char *(*))(&data), NULL, (const unsigned char *(*))(&label_onehot), &n_samples, "train", &dataset, n_train, args.batch_size, n_iters));
            if (data == NULL) { // 训练集全部使用了一轮, 当前epoch结束
                fprintf(stdout, "n_iters = %d, data is NULL\n", n_iters);
                break;
            }
            CHK_ERR(transformToFloat32FromUint8(input, data, args.batch_size * MNIST_WIDTH * MNIST_HEIGHT, 255)); // 图像数据需要做数据类型转换，类标数据不需要

            // 训练
            CHK_ERR(forwardNetwork(net, input, n_samples, "float32", &args, &probe));
            CHK_ERR(backwardNetwork(net, label_onehot, n_samples, "uint8", &args, &probe));
            CHK_ERR(updateNetwork(net, &args, &probe));

            ++n_iters;
            fprintf(stdout, "finish n_iter = %d, ce_cost = %f\n", n_iters, probe.ce_cost);
            //break;
        }
        CHK_ERR(gettimeofday(&t_epoch_1, NULL));
        timersub(&t_epoch_1, &t_epoch_0, &t_epoch_2);
        fprintf(stdout, "epoch %d finish, n_iters = %d, ce_cod = %f, time elapsed: %lu.%06lus\n", n_epochs, n_iters, probe.ce_cost, t_epoch_2.tv_sec, t_epoch_2.tv_usec);
    }
    CHK_ERR(gettimeofday(&t_train_1, NULL));
    timersub(&t_train_1, &t_train_0, &t_train_2);
    fprintf(stdout, "train finish, total epochs = %d, time elapsed: %lu.%06lus\n", n_epochs, t_train_2.tv_sec, t_train_2.tv_usec);

    // 资源释放
    destroyNetwork(net);
    printf("network destroy finish.\n");

    destroyLinearLayer(linear_0);
    destroySigmoidLayer(sigmoid_0);
    destroyLinearLayer(linear_1);
    destroyCECost(ce_cost);
    printf("layers destroy finish.\n");

    free(input);
    freeMnist(&dataset);
    free(probe.p_class);
    printf("dataset free finish\n");

    printf("all finish.\n");

    return 0;
}