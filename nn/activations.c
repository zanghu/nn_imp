#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "activations.h"

char *getActivationStrFromEnum(enum ActivationType a)
{
    switch(a){
        case LOGISTIC:
        return "logistic";
        case LOGGY:
        return "loggy";
        case RELU:
        return "relu";
        case ELU:
        return "elu";
        case SELU:
        return "selu";
        case RELIE:
        return "relie";
        case RAMP:
        return "ramp";
        case LINEAR:
        return "linear";
        case TANH:
        return "tanh";
        case PLSE:
        return "plse";
        case LEAKY:
        return "leaky";
        case STAIR:
        return "stair";
        case HARDTAN:
        return "hardtan";
        case LHTAN:
        return "lhtan";
        default:
        ERR_MSG("Unknow Activation Type, error.\n");
        return NULL;
    }
    return NULL;
}

enum ActivationType getActivationEnumFromStr(const char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu")==0) return SELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    ERR_MSG( "Couldn't find activation function %s, error.\n", s);
    return ACT_UNKNOW;
}

float runActivation(float x, enum ActivationType a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        // ReLU的一种变形, 当输入信号小于0时, 不会被全部截断, 而是被乘以一个<1的系数后继续传播
        case LEAKY:
            return leaky_activate(x); // activations.h, inline函数定义在头文件中, inline float leaky_activate(float x){return (x>0) ? x : .1*x;}
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}
