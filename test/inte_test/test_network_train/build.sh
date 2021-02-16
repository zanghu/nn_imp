#!/bin/bash

set -ex

PROJECT_DIR="../../.."

SRC_DIR="$PROJECT_DIR/src"
TEST_DIR="$PROJECT_DIR/test"

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

INC_CMD="-I. -I$SRC_DIR -I$SRC_DIR/datasets"
LIB_CMD="-lm"
#CFLAGS="-g -Wall -O2 -fopenmp"
CFLAGS="-g -Wall -O2"

gcc $CFLAGS \
    $INC_CMD \
    test.c \
    $SRC_DIR/datasets/mnist.c \
    $SRC_DIR/datasets/data_utils.c \
    $SRC_DIR/network.c \
    $SRC_DIR/layer.c \
    $SRC_DIR/linear_layer.c \
    $SRC_DIR/sigmoid_layer.c \
    $SRC_DIR/softmax_layer.c \
    $SRC_DIR/cost.c \
    $SRC_DIR/ce_cost.c \
    $SRC_DIR/opt_alg.c \
    $SRC_DIR/tensor.c \
    $SRC_DIR/gemm.c \
    $SRC_DIR/math_utils.c \
    $SRC_DIR/debug_macros.c \
    $LIB_CMD \
    -o Test