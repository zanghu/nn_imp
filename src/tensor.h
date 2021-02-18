#pragma once

#include "activations.h"

enum DType
{
    UNKNOW_DTYPE,
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    UINT8
};

const char *getTensorDtypeStrFromEnum(enum DType dtype);
enum DType getTensorDtypeEnumFromStr(const char *dtype_str);

struct Tensor;

int createTensor(struct Tensor **t, int batch_size, int row, int col, int channel);
int createTensorI32(struct Tensor **t, int batch_size, int row, int col, int channel);
int createTensorWithDataRef(struct Tensor **t, int n_samples, int row, int col, int c, const void *data, enum DType dtype);
void destroyTensor(struct Tensor *tensor);
int copyTensorData(void *dst, enum DType dtype, const struct Tensor *tensor);
void initTensorParameterAsWeight(struct Tensor *tensor);
void initTensorParameterAsBias(struct Tensor *tensor);

int getTensorShape(int *b, int *row, int *col, int *c, const struct Tensor *tensor);
int getTensorRowAndCol(int *row, int *col, const struct Tensor *tensor);
int getTensorBatchAndChannel(int *b, int *c, const struct Tensor *tensor);
int getTensorBatch(int *b, const struct Tensor *tensor);
int getTensorRow(int *row, const struct Tensor *tensor);
int getTensorCol(int *col, const struct Tensor *tensor);
int getTensorChannel(int *c, const struct Tensor *tensor);
int getTensorSamples(int *n, const struct Tensor *tensor);
int getTensorDType(enum DType *dtype, const struct Tensor *tensor);
int getTensorData(void **data, struct Tensor *tensor);
int setTensorData(struct Tensor *tensor, const void *data, enum DType dtype, int n);
int setTensorBatchAndDataByReplace(struct Tensor *tensor, const void *data, int n_samples, enum DType dtype, int need_free);

int linearTensor(struct Tensor *z, const struct Tensor *x, int xt, const struct Tensor *y, int yt, const struct Tensor *b);
int linearTensor1(struct Tensor *z, const struct Tensor *x, int xt, const struct Tensor *y, int yt, const struct Tensor *b);
int linearTensor2(struct Tensor *z, const struct Tensor *x, int xt, const struct Tensor *y, int yt, const struct Tensor *b);
int activateTensor(struct Tensor *y, const struct Tensor *x, enum ActivationType act_type);
//int deactivateTensor(const struct Tensor *delta, const struct Tensor *output, enum ActivationType act_type);
int deactivateTensor(struct Tensor *delta_out, const struct Tensor *delta_in, const struct Tensor *input, enum ActivationType act_type);
//int mulTensor(struct Tensor *z, const struct Tensor *x, const struct Tensor *y);
//int mulTensorPointwiseAndSum(struct Tensor *z, const struct Tensor *x, const struct Tensor *y);
int sumTensorAxisCol(struct Tensor *z, const struct Tensor *x);
int addTensor(struct Tensor *x, struct Tensor *y, float lr, float momentum);
int softmaxTensor(struct Tensor *output, const struct Tensor *input);
int addTensor2(struct Tensor *delta, const struct Tensor *y, const struct Tensor *gt);
int probTensor(float *val, const struct Tensor *p, const struct Tensor *gt);

int savetxtTensorData(const struct Tensor *t, const char *dst_dir, const char *prefix, const char *name, int n_epoch, int n_iter);
int savetxtTensorParam(const struct Tensor *t, const char *dst_dir, const char *prefix, const char *name, int n_epoch, int n_iter);

int openTensorLog(const char *log_path);
int closeTensorLog();
void logTensorParam(const struct Tensor *t);
void logTensorData(const struct Tensor *t);
void logTensorStr(const char *str);