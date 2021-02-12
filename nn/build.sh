#!/bin/bash

set -ex

gcc -g -Wall \
    -fPIC -shared \
    network.c \
    layer.c \
    matrix.c \
    math_utils.c \
    debug_macros.c \
    -o libnn.so