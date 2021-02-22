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

enum TensorType
{
    UNKNOW_TENSOR_TYPE,
    DATA_TENSOR_TYPE,
    PARAM_TENSOR_TYPE
};

const char *getTensorDtypeStrFromEnum(enum DType dtype);
enum DType getTensorDtypeEnumFromStr(const char *dtype_str);
const char *getTensorTtypeStrFromEnum(enum TensorType ttype);

struct Tensor;

int initTensorParameterAsWeight(struct Tensor *tensor);
int initTensorParameterAsBias(struct Tensor *tensor);

int createTensorData(struct Tensor **t, enum DType dtype, int batch_size, int n_features);
int createTensorParam(struct Tensor **t, enum DType dtype, int row, int col);
int createTensorDataWithBlobRef(struct Tensor **t, void *blob, enum DType dtype, int batch_size, int n_features, int n_samples);
void destroyTensor(struct Tensor *tensor);
int getTensorBlobByCopy(void *dst, enum DType dtype, const struct Tensor *tensor);
int getTensorRowAndCol(int *row, int *col, const struct Tensor *tensor);
int getTensorBatchAndFeatures(int *batch_size, int *n_features, const struct Tensor *tensor);
int getTensorRow(int *row, const struct Tensor *tensor);
int getTensorCol(int *col, const struct Tensor *tensor);
int getTensorBatch(int *b, const struct Tensor *tensor);
int getTensorFeatures(int *n_features, const struct Tensor *tensor);
int getTensorSamples(int *n_samples, const struct Tensor *tensor);
int getTensorDType(enum DType *dtype, const struct Tensor *tensor);
int getTensorType(enum TensorType *ttype, const struct Tensor *tensor);
int getTensorBlob(void **blob, struct Tensor *tensor);
int setTensorSamplesByReplace(void **blob_old, struct Tensor *tensor, void *blob, int n_samples, int n_features, enum DType dtype);

int activateTensor(struct Tensor *y, const struct Tensor *x, enum ActivationType act_type);
int deactivateTensor(struct Tensor *delta_out, const struct Tensor *delta_in, const struct Tensor *input, enum ActivationType act_type);
int linearTensorForward(struct Tensor *z, const struct Tensor *x, const struct Tensor *y, const struct Tensor *b);
int linearTensorBackward(struct Tensor *z, const struct Tensor *x, const struct Tensor *y);
int linearTensorWeightGradient(struct Tensor *z, const struct Tensor *x, const struct Tensor *y);
int linearTensorBiasGradient(struct Tensor *z, const struct Tensor *x);
int addTensor(struct Tensor *x, struct Tensor *y, float lr, float momentum);
int softmaxTensor(struct Tensor *output, const struct Tensor *input);
int addTensor2(struct Tensor *delta, const struct Tensor *y, const struct Tensor *gt);
int probTensor(float *val, const struct Tensor *p, const struct Tensor *gt);

int savetxtTensorData(const struct Tensor *t, const char *dst_dir, const char *prefix, const char *name, int n_epoch, int n_iter);
int savetxtTensorParam(const struct Tensor *t, const char *dst_dir, const char *prefix, const char *name, int n_epoch, int n_iter);

/*
int openTensorLog(const char *log_path);
int closeTensorLog();
void logTensorParam(const struct Tensor *t);
void logTensorData(const struct Tensor *t);
void logTensorStr(const char *str);
*/