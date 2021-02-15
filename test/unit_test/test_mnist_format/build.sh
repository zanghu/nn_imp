#!/bin/bash

set -ex

SRC_DIR=../../../src

INC_CMD="-I$SRC_DIR -I$SRC_DIR/datasets"

gcc -g -Wall $INC_CMD test.c $SRC_DIR/datasets/mnist.c $SRC_DIR/debug_macros.c -o Test