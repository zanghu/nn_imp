#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "network.h"
#include "layer.h"
#include "linear_layer.h"
#include "sigmoid_layer.h"
#include "cost.h"
#include "ce_cost.h"
#include "opt_alg.h"
#include "debug_macros.h"

int main()
{
    struct Network *net = NULL;

    struct LinearLayer *linear_0 = NULL;
    CHK_ERR(createLinearLayer(&linear_0, 28 * 28, 625));
    struct SigmoidLayer *sigmoid_0 = NULL;
    CHK_ERR(createSigmoidLayer(&sigmoid_0, 625));
    struct LinearLayer *linear_1 = NULL;
    CHK_ERR(createLinearLayer(&linear_0, 625, 10));
    struct CECost *ce_cost = NULL;
    CHK_ERR(createCECost(&ce_cost, 10));

    struct Layer *layers[] = {(struct Layer *)linear_0, (struct Layer *)sigmoid_0, (struct Layer *)linear_1};
    struct UpdateArgs args;
    args.batch_size = 128;
    args.lr = 0.05;
    args.momentum = 0.0; // 不使用动量法
    CHK_ERR(createNetwork(&net, layers, 3, (struct Cost *)ce_cost, &args));

    return 0;
}