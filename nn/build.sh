#!/bin/bash

set -ex

#gcc -g -Wall -O2 \
#    -fPIC -shared \
#    -fopenmp \
#    network.c \
#    layer.c \
#    linear_layer.c \
#    sigmoid_layer.c \
#    cost.c \
#    ce_cost.c \
#    opt_alg.c \
#    tensor.c \
#    gemm.c \
#    math_utils.c \
#    debug_macros.c \
#    -o libnn.so

gcc -g -Wall -O2 \
    -fopenmp \
    test.c \
    network.c \
    layer.c \
    linear_layer.c \
    sigmoid_layer.c \
    softmax_layer.c \
    cost.c \
    ce_cost.c \
    opt_alg.c \
    tensor.c \
    gemm.c \
    math_utils.c \
    debug_macros.c \
    -lm \
    -o Test