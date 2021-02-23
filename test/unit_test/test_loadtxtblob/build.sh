#!/bin/bash

set -ex

SRC_DIR=../../../src

INC_CMD="-I$SRC_DIR"

gcc -g -Wall $INC_CMD test.c $SRC_DIR/io_utils.c $SRC_DIR/debug_macros.c -o Test