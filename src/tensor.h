#pragma once

#include "activations.h"

enum DType
{
    FLOAT32,
    FLOAT64,
    INT32,
    INT64
};

struct Tensor;

int createTensor(struct Tensor **t, int batch_size, int row, int col, int channel);
int createTensorI32(struct Tensor **t, int batch_size, int row, int col, int channel);
void destroyTensor(struct Tensor *matrix);

void initTensorParameterAsWeight(struct Tensor *matrix);
void initTensorParameterAsBias(struct Tensor *matrix);

int getTensorRow(int *row, const struct Tensor *matrix);
int getTensorCol(int *col, const struct Tensor *matrix);
int getTensorChannel(int *c, const struct Tensor *matrix);
int getTensorBatch(int *b, const struct Tensor *matrix);
int getTensorDType(enum DType *dtype, const struct Tensor *matrix);
int getTensorData(void **data, struct Tensor *tensor);

int linearTensor(struct Tensor *z, const struct Tensor *x, int xt, const struct Tensor *y, int yt, const struct Tensor *b);
int activateTensor(struct Tensor *y, const struct Tensor *x, enum ActivationType act_type);
//int deactivateTensor(const struct Tensor *delta, const struct Tensor *output, enum ActivationType act_type);
int deactivateTensor(struct Tensor *delta_out, const struct Tensor *delta_in, const struct Tensor *input, enum ActivationType act_type);
//int mulTensor(struct Tensor *z, const struct Tensor *x, const struct Tensor *y);
int mulTensorPointwiseAndSum(struct Tensor *z, const struct Tensor *x, const struct Tensor *y);
int addTensor(struct Tensor *x, struct Tensor *y, float lr, float momentum);
int softmaxTensor(struct Tensor *output, const struct Tensor *input);
int addTensor2(struct Tensor *delta, const struct Tensor *y, const struct Tensor *gt);
int probTensor(float *val, const struct Tensor *p, const struct Tensor *gt);