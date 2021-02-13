#!/bin/bash

set -ex

gcc -g -Wall -O2 \
    -fPIC -shared \
    -fopenmp \
    network.c \
    fc_layer.c \
    matrix.c \
    gemm.c \
    math_utils.c \
    debug_macros.c \
    -o libnn.so