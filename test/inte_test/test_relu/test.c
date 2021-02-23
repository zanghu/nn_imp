#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "network.h"
#include "layer.h"
#include "linear_layer.h"
#include "relu_layer.h"
#include "cost.h"
#include "ce_cost.h"
#include "opt_alg.h"
#include "probe.h"
#include "mnist.h"
#include "tensor.h"
#include "debug_macros.h"

#define DATASET_DIR ("/home/zanghu/data_base/mnist")

int main()
{
    //CHK_ERR(openTensorLog("log.txt"));
    int n_features = MNIST_WIDTH * MNIST_HEIGHT;
    int n_classes = MNIST_N_CLASSES;

    // 创建组件
    struct LinearLayer *linear_0 = NULL;
    CHK_ERR(createLinearLayer(&linear_0, "LIN_L0", n_features, 256));
    struct ReluLayer *relu_0 = NULL;
    CHK_ERR(createReluLayer(&relu_0, "RELU_L0"));
    struct LinearLayer *linear_1 = NULL;
    CHK_ERR(createLinearLayer(&linear_1, "LIN_L1", 256, 128));
    struct ReluLayer *relu_1 = NULL;
    CHK_ERR(createReluLayer(&relu_1, "RELU_L1"));
    struct LinearLayer *linear_2 = NULL;
    CHK_ERR(createLinearLayer(&linear_2, "LIN_L2", 128, n_classes));
    struct CECost *ce_cost = NULL;
    CHK_ERR(createCECost(&ce_cost, "CE_COST", 10));

    // 加载参数
    CHK_ERR(loadtxtLinearLayerWeight(linear_0, "txt/NET_L00_W_784x256.txt"));
    CHK_ERR(loadtxtLinearLayerBias(linear_0, "txt/NET_L00_b_256.txt"));
    CHK_ERR(loadtxtLinearLayerWeight(linear_1, "txt/NET_L02_W_256x128.txt"));
    CHK_ERR(loadtxtLinearLayerBias(linear_1, "txt/NET_L02_b_128.txt"));
    CHK_ERR(loadtxtLinearLayerWeight(linear_2, "txt/NET_L04_W_128x10.txt"));
    CHK_ERR(loadtxtLinearLayerBias(linear_2, "txt/NET_L04_b_10.txt"));

    printf("layers create finish.\n");

    // 创建网络
    struct Layer *layers[] = {(struct Layer *)linear_0, (struct Layer *)relu_0, (struct Layer *)linear_1, (struct Layer *)relu_1, (struct Layer *)linear_2};
    struct UpdateArgs args;
    args.batch_size = 128;
    args.lr = 0.001;
    args.momentum = 0.0; // 不使用动量法
    args.n_epochs = 100; // 最大循环数
    struct Network *net = NULL;
    CHK_ERR(createNetwork(&net, layers, 5, (struct Cost *)ce_cost));
    printf("network create finish.\n");

    // 读取数据集
    struct MNIST mnist;
    CHK_ERR(loadMnist(&mnist, DATASET_DIR));

    // 训练
    int n_train = 50000;
    //int n_valid = 10000;
    //unsigned char *gt = NULL;

    // 探针
    struct Probe probe;
    memset(&probe, 0, sizeof(struct Probe));
    probe.sw_p_class = 1;
    CHK_NIL((probe.p_class = calloc(args.batch_size * n_classes, sizeof(float))));
    probe.sw_ce_cost = 1;

    probe.dst_dir = "txt";
    //probe.dump_w = 1; // 导出层的参数
    //probe.dump_b = 1; // 导出层的参数
    //probe.dump_gw = 1; // 导出层的梯度
    //probe.dump_gb = 1; // 导出层的梯度
    //probe.dump_output = 1; // 导出层的输出
    //probe.dump_delta = 1; // 导出层的灵敏度

    struct timeval t_train_0, t_train_1, t_train_2;
    CHK_ERR(gettimeofday(&t_train_0, NULL));
    int n_epochs = 50;
    for (int k = 0; k < n_epochs; ++k) {
        int n_iters = 0;
        args.cur_epoch = k;
        const void *data_batch = NULL;
        const void *label_onehot = NULL;
        int n_samples = 0;
        struct timeval t_epoch_0, t_epoch_1, t_epoch_2;
        CHK_ERR(gettimeofday(&t_epoch_0, NULL));
        while (1) {
            args.cur_iter = n_iters;
            // 获取batch数据
            CHK_ERR(getMnistNthBatch((const float *(*))(&data_batch), (const unsigned char *(*))(&label_onehot), &n_samples, "train", &mnist, n_train, args.batch_size, n_iters));
            if (data_batch == NULL) { // 训练集全部使用了一轮, 当前epoch结束
                fprintf(stdout, "n_iters = %d, data_batch is NULL\n", n_iters);
                break;
            }

            // 训练
            CHK_ERR(forwardNetwork(net, data_batch, n_samples, 28 * 28, "float32", &args, &probe));
            CHK_ERR(backwardNetwork(net, label_onehot, n_samples, 10, "uint8", &args, &probe));
            CHK_ERR(updateNetwork(net, &args, &probe));

            ++n_iters;
            fprintf(stdout, "finish n_iter = %d, ce_cost = %f\n", n_iters, probe.ce_cost);
            //if (n_iters == 200) {
            //    break;
            //}
        }
        CHK_ERR(gettimeofday(&t_epoch_1, NULL));
        timersub(&t_epoch_1, &t_epoch_0, &t_epoch_2);
        fprintf(stdout, "epoch %d finish, n_iters = %d, ce_cod = %f, time elapsed: %lu.%06lus\n", k, n_iters, probe.ce_cost, t_epoch_2.tv_sec, t_epoch_2.tv_usec);
    }
    CHK_ERR(gettimeofday(&t_train_1, NULL));
    timersub(&t_train_1, &t_train_0, &t_train_2);
    fprintf(stdout, "train finish, total epochs = %d, time elapsed: %lu.%06lus\n", n_epochs, t_train_2.tv_sec, t_train_2.tv_usec);

    // 资源释放
    freeMnist(&mnist);
    free(probe.p_class);
    fprintf(stdout, "mnist free finish\n");

    destroyNetwork(net);
    fprintf(stdout, "network destroy finish.\n");

    destroyLinearLayer(linear_0);
    destroyReluLayer(relu_0);
    destroyLinearLayer(linear_1);
    destroyReluLayer(relu_1);
    destroyLinearLayer(linear_2);
    destroyCECost(ce_cost);
    fprintf(stdout, "layers destroy finish.\n");

    //CHK_ERR(closeTensorLog());
    fprintf(stdout, "all finish.\n");

    return 0;
}