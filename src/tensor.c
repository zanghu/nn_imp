#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "debug_macros.h"
#include "math_utils.h"
#include "activations.h"
#include "gemm.h"
#include "tensor.h"
#include "io_utils.h"
#include "const.h"

/*
static FILE *g_fp = NULL;

int openTensorLog(const char *log_path)
{
    CHK_NIL(log_path);
    g_fp = fopen(log_path, "wb");
    if (g_fp == NULL) {
        ERR_MSG("fopen failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    fprintf(stdout, "g_fp open success.\n");
    return SUCCESS;
}

int closeTensorLog()
{
    if (fclose(g_fp) == EOF) {
        ERR_MSG("fclose failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    return SUCCESS;
}
*/

struct Tensor {
    enum TensorType ttype;
    enum DType dtype;

    // DATA_TENSOR_TYPE only
    int b; // batch_size
    int n; // n_features
    int b_used; // 当前实际装填的样本数 b_used <= b

    // PARAM_TENSOR_TYPE only
    int row;
    int col;

    // blob 
    float *blob;
    double *blob_f64;
    int *blob_i32;
    long long *blob_i64;
    unsigned char *blob_u8;
};

const char *getTensorDtypeStrFromEnum(enum DType dtype)
{
    switch (dtype) {
        case FLOAT32:
        return "float32";

        case FLOAT64:
        return "float64";

        case INT32:
        return "int32";

        case INT64:
        return "int64";

        case UINT8:
        return "uint8";

        default:
        return "unknow dtype";
    }
    return "unknow dtype";
}

enum DType getTensorDtypeEnumFromStr(const char *dtype_str)
{
    if (strcasecmp("float32", dtype_str) == 0) {
        return FLOAT32;
    } else if(strcasecmp("float64", dtype_str) == 0) {
        return FLOAT64;
    } else if(strcasecmp("int32", dtype_str) == 0) {
        return INT32;
    } else if(strcasecmp("int64", dtype_str) == 0) {
        return INT64;
    } else if(strcasecmp("uint8", dtype_str) == 0) {
        return UINT8;
    }

    return UNKNOW_DTYPE;
}

const char *getTensorTtypeStrFromEnum(enum TensorType ttype)
{
    switch (ttype) {
        case DATA_TENSOR_TYPE:
        return "data_tensor_type";

        case PARAM_TENSOR_TYPE:
        return "param_tensor_type";

        default:
        return "unknow_tensor_type";
    }
    return "unknow_tensor_type";
}

int initTensorParameterAsWeight(struct Tensor *tensor)
{
    CHK_NIL(tensor);
    CHK_ERR((tensor->ttype == PARAM_TENSOR_TYPE)? 0: 1);
    int inputs = tensor->row;
    float scale = sqrt(2. / inputs);
    int i = 0;
    for(i = 0; i < tensor->row * tensor->col; ++i){
        tensor->blob[i] = scale * rand_uniform(-1, 1);
    }
    return SUCCESS;
}

int initTensorParameterAsBias(struct Tensor *tensor)
{
    CHK_NIL(tensor);
    CHK_ERR((tensor->ttype == PARAM_TENSOR_TYPE)? 0: 1);

    memset(tensor->blob, 0, tensor->row * tensor->col * sizeof(float));

    return SUCCESS;
}

int createTensorData(struct Tensor **t, enum DType dtype, int batch_size, int n_features)
{
    CHK_NIL(t);
    CHK_ERR((batch_size > 0)? 0: 1);
    CHK_ERR((n_features > 0)? 0: 1);

    struct Tensor *tensor = calloc(1, sizeof(struct Tensor));
    if (!tensor) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
 
    switch (dtype) {
        case FLOAT32:
        tensor->blob = calloc(batch_size * n_features, sizeof(float));
        if (tensor->blob == NULL) {
            ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
            goto err_end;
        }
        break;

        case UINT8:
        tensor->blob = calloc(batch_size * n_features, sizeof(unsigned char));
        if (tensor->blob_u8 == NULL) {
            ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
            goto err_end;
        }
        break;

        default:
        ERR_MSG("DType: %s is not supported by DATA Tensor yet, error.\n", getTensorDtypeStrFromEnum(dtype));
        goto err_end;
    }
    tensor->ttype = DATA_TENSOR_TYPE;
    tensor->dtype = dtype;
    tensor->b = batch_size;
    tensor->n = n_features;
    tensor->b_used = 0; // 初始装填样本数为0

    *t = tensor;
    return SUCCESS;

err_end:
    if (tensor) {
        free(tensor->blob);
        free(tensor->blob_u8);
        free(tensor->blob_f64);
        free(tensor->blob_i32);
        free(tensor->blob_i64);
    }
    free(tensor);
    return ERR_COD;
}

int createTensorParam(struct Tensor **t, enum DType dtype, int row, int col)
{
    CHK_NIL(t);
    CHK_ERR((row > 0)? 0: 1);
    CHK_ERR((col > 0)? 0: 1);

    struct Tensor *tensor = calloc(1, sizeof(struct Tensor));
    if (!tensor) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
 
    switch (dtype) {
        case FLOAT32: // 目前参数的数据类型只支持32位浮点数float，这也是现阶段GPU计算的需要
        tensor->blob = calloc(row * col, sizeof(float));
        if (tensor->blob == NULL) {
            ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
            goto err_end;
        }
        break;

        default:
        ERR_MSG("DType: %s is not supported by PARAM Tensor yet, error.\n", getTensorDtypeStrFromEnum(dtype));
        goto err_end;
    }
    tensor->ttype = PARAM_TENSOR_TYPE;
    tensor->dtype = dtype;
    tensor->row = row;
    tensor->col = col;

    *t = tensor;
    return SUCCESS;

err_end:
    if (tensor) {
        free(tensor->blob);
        free(tensor->blob_u8);
        free(tensor->blob_f64);
        free(tensor->blob_i32);
        free(tensor->blob_i64);
    }
    free(tensor);
    return ERR_COD;
}

int createTensorDataWithBlobRef(struct Tensor **t, void *blob, enum DType dtype, int batch_size, int n_features, int n_samples)
{
    CHK_NIL(t);
    CHK_NIL(blob);
    CHK_ERR((batch_size > 0)? 0: 1);
    CHK_ERR((n_features > 0)? 0: 1);
    CHK_ERR((n_samples > 0)? 0: 1);

    struct Tensor *tensor = calloc(1, sizeof(struct Tensor));
    if (!tensor) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    switch (dtype) {
        case FLOAT32:
        tensor->blob = (float *)blob;
        break;

        case UINT8:
        tensor->blob_u8 = (unsigned char *)blob;
        break;

        default:
        ERR_MSG("DType: %s is not supported yet, error.\n", getTensorDtypeStrFromEnum(dtype));
        free(tensor);
        return ERR_COD;
    }
    tensor->ttype = DATA_TENSOR_TYPE;
    tensor->dtype = dtype;
    tensor->b = batch_size;
    tensor->b_used = n_samples;
    tensor->n = n_features;

    *t = tensor;
    return SUCCESS;
}

void destroyTensor(struct Tensor *tensor)
{
    if (tensor) {
        free(tensor->blob);
        free(tensor->blob_f64);
        free(tensor->blob_i32);
        free(tensor->blob_i64);
        free(tensor->blob_u8);
    }
    free(tensor);
}

int loadtxtTensor(struct Tensor *tensor, const char *pth)
{
    CHK_NIL(tensor);
    CHK_NIL(pth);

    int n_samples = 0;
    int n_features = 0;
    if (tensor->ttype == DATA_TENSOR_TYPE) {
        n_samples = tensor->b_used;
        n_features = tensor->n;
    } else if (tensor->ttype == PARAM_TENSOR_TYPE) {
        n_samples = tensor->row;
        n_features = tensor->col;
    } else {
        ERR_MSG("Unknow TensorType: %s, error.\n", getTensorTtypeStrFromEnum(tensor->ttype));
        return ERR_COD;
    }
    //fprintf(stdout, "(n_samples, n_features) = (%d, %d)\n", n_samples, n_features);

    int n_loaded = 0;
    switch (tensor->dtype) {
        case FLOAT32:
        memset(tensor->blob, 0, n_samples * n_features * sizeof(float));
        tensor->b_used = 0;
        CHK_ERR(loadtxtBlobFloat32(&n_loaded, pth, tensor->blob, n_samples * n_features));
        if (n_loaded % n_features != 0) {
            ERR_MSG("n_load = %d is not integral times of n_features = %d, error.\n", n_loaded, n_features);
            return ERR_COD;
        }
        if (tensor->ttype == DATA_TENSOR_TYPE) {
            tensor->b_used = n_loaded / n_features;
        }
        if (tensor->ttype == PARAM_TENSOR_TYPE && n_loaded != n_features * n_samples) {
            ERR_MSG("error occured.\n");
            return ERR_COD;
        }
        break;

        default:
        ERR_MSG("DType: %s not supported yet, error.\n", getTensorDtypeStrFromEnum(tensor->dtype));
        return ERR_COD;
    }
    return SUCCESS;
}

int getTensorBlobByCopy(void *dst, enum DType dtype, const struct Tensor *tensor)
{
    CHK_NIL(dst);
    CHK_NIL(tensor);
    CHK_ERR((tensor->dtype == dtype)? 0: 1);

    if (tensor->ttype == DATA_TENSOR_TYPE) {
        int n_elem = tensor->b * tensor->n;
        switch (dtype) {
            case FLOAT32:
            memcpy(dst, tensor->blob, n_elem * sizeof(float));
            break;

            case UINT8:
            memcpy(dst, tensor->blob_u8, n_elem * sizeof(unsigned char));
            break;

            default:
            ERR_MSG("DType: %s is not supported yet, error.\n", getTensorDtypeStrFromEnum(dtype));
            return ERR_COD;
        }
    }
    else if (tensor->ttype == PARAM_TENSOR_TYPE) {
        int n_elem = tensor->row * tensor->col;
        switch (dtype) {
            case FLOAT32:
            memcpy(dst, tensor->blob, n_elem * sizeof(float));
            break;

            case UINT8:
            memcpy(dst, tensor->blob_u8, n_elem * sizeof(unsigned char));
            break;

            default:
            ERR_MSG("DType: %s is not supported yet, error.\n", getTensorDtypeStrFromEnum(dtype));
            return ERR_COD;
        }
    } else {
        ERR_MSG("TTYPE: %s is not supported yet, error.\n", getTensorTtypeStrFromEnum(tensor->ttype));
        return ERR_COD;
    }

    return SUCCESS;
}

int getTensorRowAndCol(int *row, int *col, const struct Tensor *tensor)
{
    CHK_NIL(row);
    CHK_NIL(col);
    CHK_NIL(tensor);
    CHK_ERR((tensor->ttype == PARAM_TENSOR_TYPE)? 0: 1);

    *row = tensor->row;
    *col = tensor->col;
    return SUCCESS;
}

int getTensorBatchAndFeatures(int *batch_size, int *n_features, const struct Tensor *tensor)
{
    CHK_NIL(batch_size);
    CHK_NIL(n_features);
    CHK_NIL(tensor);
    CHK_ERR((tensor->ttype == DATA_TENSOR_TYPE)? 0: 1);

    *batch_size = tensor->b;
    *n_features = tensor->n;
    return SUCCESS;
}

int getTensorRow(int *row, const struct Tensor *tensor)
{
    CHK_NIL(row);
    CHK_NIL(tensor);
    CHK_ERR((tensor->ttype == PARAM_TENSOR_TYPE)? 0: 1);
    *row = tensor->row;
    return SUCCESS;
}

int getTensorCol(int *col, const struct Tensor *tensor)
{
    CHK_NIL(col);
    CHK_NIL(tensor);
    CHK_ERR((tensor->ttype == PARAM_TENSOR_TYPE)? 0: 1);
    *col = tensor->col;
    return SUCCESS;
}

int getTensorBatch(int *b, const struct Tensor *tensor)
{
    CHK_NIL(b);
    CHK_NIL(tensor);
    CHK_ERR((tensor->ttype == DATA_TENSOR_TYPE)? 0: 1);
    *b = tensor->b;
    return SUCCESS;
}

int getTensorFeatures(int *n_features, const struct Tensor *tensor)
{
    CHK_NIL(n_features);
    CHK_NIL(tensor);
    CHK_ERR((tensor->ttype == DATA_TENSOR_TYPE)? 0: 1);
    *n_features = tensor->n;
    return SUCCESS;
}

int getTensorSamples(int *n_samples, const struct Tensor *tensor)
{
    CHK_NIL(n_samples);
    CHK_NIL(tensor);
    CHK_ERR((tensor->ttype == DATA_TENSOR_TYPE)? 0: 1);
    *n_samples = tensor->b_used;
    return SUCCESS;
}

int getTensorDType(enum DType *dtype, const struct Tensor *tensor)
{
    CHK_NIL(dtype);
    CHK_NIL(tensor);
    *dtype = tensor->dtype;
    return SUCCESS;
}

int getTensorType(enum TensorType *ttype, const struct Tensor *tensor)
{
    CHK_NIL(ttype);
    CHK_NIL(tensor);
    *ttype = tensor->ttype;
    return SUCCESS;
}

int getTensorBlob(void **blob, struct Tensor *tensor)
{
    CHK_NIL(blob);
    CHK_NIL(tensor);

    switch (tensor->dtype) {
        case FLOAT32:
        *blob = tensor->blob;
        break;

        case UINT8:
        *blob = tensor->blob_u8;
        break;

        default:
        ERR_MSG("DType: %s is not supported yet, error.\n", getTensorDtypeStrFromEnum(tensor->dtype));
        return ERR_COD;
    }
    return SUCCESS;
}

int setTensorSamplesByReplace(void **blob_old, struct Tensor *tensor, void *blob, int n_samples, int n_features, enum DType dtype)
{
    CHK_NIL(blob_old);
    CHK_NIL(tensor);
    CHK_NIL(blob);
    CHK_ERR((n_samples > 0)? 0: 1);
    CHK_ERR((n_features > 0)? 0: 1);
    CHK_ERR((tensor->dtype == dtype)? 0: 1);
    CHK_ERR((tensor->ttype == DATA_TENSOR_TYPE)? 0: 1);
    //fprintf(stdout, "n_samples = %d, tensor->b = %d\n", n_samples, tensor->b);
    //CHK_ERR((n_samples <= tensor->b)? 0: 1); 此项无需检查，因为是直接替换blob而不是向blob拷贝，加入该检查后回到值epoch切换时报错（因为上一个epoch最后一个batch缩小了blob）
    CHK_ERR((n_features == tensor->n)? 0: 1);

    void *tmp = NULL;
    switch (dtype) {
        case UINT8:
        tmp = tensor->blob_u8;
        tensor->blob_u8 = (unsigned char *)blob;
        break;

        case FLOAT32:
        tmp = tensor->blob;
        tensor->blob = (float *)blob;
        break;

        default:
        ERR_MSG("DType: %s is not supported yet, error.\n", getTensorDtypeStrFromEnum(dtype));
        return ERR_COD;
    }
    *blob_old = tmp;
    tensor->b_used = n_samples;
    tensor->b = n_samples;
    return SUCCESS;
}

// 前向传播过程的非线性运算部分
int activateTensor(struct Tensor *y, const struct Tensor *x, enum ActivationType act_type)
{
    CHK_NIL(x);
    CHK_ERR((x->n > 0)? 0: 1);
    CHK_ERR((x->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((x->ttype == y->ttype)? 0: 1);

    if (x->ttype == DATA_TENSOR_TYPE) {
        int i = 0;
        switch (act_type) {
            case LOGISTIC:
            //fprintf(stdout, "x->b_used = %d,  x->n = %d\n", x->b_used, x->n);
            for (i = 0; i < x->b_used * x->n; ++i) {
                y->blob[i] = logistic_activate(x->blob[i]); // activations.h, inline
            }
            break;

            case RELU:
            //fprintf(stdout, "x->b_used = %d,  x->n = %d\n", x->b_used, x->n);
            for (i = 0; i < x->b_used * x->n; ++i) {
                y->blob[i] = relu_activate(x->blob[i]); // activations.h, inline
            }
            break;

            default:
            ERR_MSG("Unknow activation_type, error.\n");
            return ERR_COD;
        }
        y->b_used = x->b_used;
    }
    else {
        ERR_MSG("TensorType: %s not supported yet, error.\n", getTensorTtypeStrFromEnum(x->ttype));
        return ERR_COD;
    }
    return SUCCESS;
}

// 反向传播过程的非线性运算部分, 计算后的结果会替换输入值delta的原有内容
int deactivateTensor(struct Tensor *delta_out, const struct Tensor *delta_in, const struct Tensor *input, enum ActivationType act_type)
{
    CHK_NIL(delta_out);
    CHK_NIL(delta_in);
    CHK_NIL(input);
    CHK_ERR((delta_out->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((delta_in->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((input->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((delta_in->n > 0)? 0: 1);
    CHK_ERR((delta_in->n == input->n)? 0: 1);
    CHK_ERR((delta_in->n == delta_out->n)? 0: 1);
    CHK_ERR((delta_in->b_used > 0)? 0: 1);
    CHK_ERR((delta_in->b_used == input->b_used)? 0: 1);

    int i = 0;
    switch (act_type) {
        case LOGISTIC:
        for (i = 0; i < input->b_used * input->n; ++i) {
            delta_out->blob[i] = logistic_gradient(input->blob[i]) * delta_in->blob[i]; // 注意这里的x应该是outputs[i]，而不是hiddens[i]
        }
        break;

        case RELU:
        for (i = 0; i < input->b_used * input->n; ++i) {
            delta_out->blob[i] = relu_gradient(input->blob[i]) * delta_in->blob[i]; // 注意这里的x应该是outputs[i]，而不是hiddens[i]
        }
        break;

        default:
        ERR_MSG("Unknow dectivation_type, error.\n");
        return ERR_COD;
    }
    delta_out->b_used = delta_in->b_used;
    return SUCCESS;
}

/**
 * @param x: 输入参数，shape = (batch_size, n_input)
 * @param y: 输入参数，shape = (barch_size, n_output)
 * @param z: 输出参数, shape = (n_input, n_output)
 * @param b: 输入参数, shape = (1, n_output)
 */
int linearTensorForward(struct Tensor *z, const struct Tensor *x, const struct Tensor *y, const struct Tensor *b)
{
    CHK_NIL(z);
    CHK_NIL(x);
    CHK_NIL(y);
    CHK_ERR((x->b_used > 0)? 0: 1);
    CHK_ERR((x->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((z->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((y->ttype == PARAM_TENSOR_TYPE)? 0: 1);
    CHK_ERR((b->ttype == PARAM_TENSOR_TYPE)? 0: 1);
    CHK_ERR((x->n == y->col)? 0: 1);
    CHK_ERR((z->n == y->row)? 0: 1);
#ifdef _DEBUG
    fprintf(stdout, "(z.b, z.row, z.col, z.c, z.n) = (%d, %d, %d, %d, %d)\n", z->b, z->row, z->col, z->c, z->n);
    fprintf(stdout, "(y.b, y.row, y.col, y.c, y.n) = (%d, %d, %d, %d, %d)\n", y->b, y->row, y->col, y->c, y->n);
    fprintf(stdout, "(x.b, x.row, x.col, x.c, x.n) = (%d, %d, %d, %d, %d)\n", x->b, x->row, x->col, x->c, x->n);
#endif

    if (b) {
        int i = 0;
        for (i = 0; i < x->b_used; ++i) {
            memcpy(z->blob + i * z->n, b->blob, sizeof(float) * b->col);
        }
    }

    // gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
    //     float *A, int lda, 
    //     float *B, int ldb,
    //     float BETA,
    //     float *C, int ldc)
    gemm(0, 1, x->b_used, y->row, y->col, 1., 
        x->blob, x->n, 
        y->blob, y->col, 
        1., 
        z->blob, z->n);
    z->b_used = x->b_used;
    return SUCCESS;
}

int linearTensorBackward(struct Tensor *z, const struct Tensor *x, const struct Tensor *y)
{
    CHK_NIL(z);
    CHK_NIL(x);
    CHK_NIL(y);
    CHK_ERR((x->b_used > 0)? 0: 1);
    CHK_ERR((x->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((z->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((y->ttype == PARAM_TENSOR_TYPE)? 0: 1);
    CHK_ERR((z->n == y->col)? 0: 1);
    CHK_ERR((x->n == y->row)? 0: 1);
#ifdef _DEBUG
    fprintf(stdout, "(z.b, z.row, z.col, z.c, z.n) = (%d, %d, %d, %d, %d)\n", z->b, z->row, z->col, z->c, z->n);
    fprintf(stdout, "(y.b, y.row, y.col, y.c, y.n) = (%d, %d, %d, %d, %d)\n", y->b, y->row, y->col, y->c, y->n);
    fprintf(stdout, "(x.b, x.row, x.col, x.c, x.n) = (%d, %d, %d, %d, %d)\n", x->b, x->row, x->col, x->c, x->n);
#endif

    // gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
    //     float *A, int lda, 
    //     float *B, int ldb,
    //     float BETA,
    //     float *C, int ldc)
    gemm(0, 0, x->b_used, y->col, y->row, 1., 
        x->blob, x->n, 
        y->blob, y->col, 
        1., 
        z->blob, z->n);
    z->b_used = x->b_used;
    return SUCCESS;
}

int linearTensorWeightGradient(struct Tensor *z, const struct Tensor *x, const struct Tensor *y)
{
    CHK_NIL(z);
    CHK_NIL(x);
    CHK_NIL(y);
    CHK_ERR((x->b_used > 0)? 0: 1);
    CHK_ERR((x->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((y->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((z->ttype == PARAM_TENSOR_TYPE)? 0: 1);
    CHK_ERR((z->col == y->n)? 0: 1);
    CHK_ERR((z->row == x->n)? 0: 1);
    CHK_ERR((x->b_used == y->b_used)? 0: 1);
#ifdef _DEBUG
    fprintf(stdout, "(z.r, z.c) = (%d, %d)\n", z->row, z->col);
    fprintf(stdout, "(y.b, y.n) = (%d, %d)\n", y->b_used, y->n);
    fprintf(stdout, "(x.b, x.n) = (%d, %d)\n", x->b_used, x->n);
    fprintf(stdout, "x->blob[128] = %f\n", x->blob[128]);
#endif
    // gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
    //     float *A, int lda, 
    //     float *B, int ldb,
    //     float BETA,
    //     float *C, int ldc)
    //fprintf(stdout, "start linearTensorWeightGradient++++++++++\n");
    gemm(1, 0, x->n, y->n, x->b_used, 1., 
        x->blob, x->n, 
        y->blob, y->n, 
        1., 
        z->blob, z->col);
    //fprintf(stdout, "finish linearTensorWeightGradient----------\n");
    return SUCCESS;
}


// 2个形状完全相同的矩阵x和y的元素做Pointwise乘法, 之后每行求和, 压缩成一个向量, 结果保存在z中
// 该函数用来在反向传播时计算偏置bias的梯度
//int sumTensorAxisCol(struct Tensor *z, const struct Tensor *x)
int linearTensorBiasGradient(struct Tensor *z, const struct Tensor *x)
{
    CHK_NIL(z);
    CHK_NIL(x);
    CHK_ERR((x->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((z->ttype == PARAM_TENSOR_TYPE)? 0: 1);
    CHK_ERR((z->col == x->n)? 0: 1);
#ifdef _DEBUG
    fprintf(stdout, "(x.b, x.row, x.col, x.c, x.n) = (%d, %d, %d, %d, %d)\n", x->b, x->row, x->col, x->c, x->n);
    fprintf(stdout, "(z.b, z.row, z.col, z.c. z.n) = (%d, %d, %d, %d, %d)\n", z->b, z->row, z->col, z->c, z->n);
#endif

    int b_used = x->b_used; // batch_size
    int n = x->n; // n_output
    int i, j;
    for (i = 0; i < n; ++i) {
        z->blob[i] = 0.;
        for (j = 0; j < b_used; ++j) {
            z->blob[i] += x->blob[j * n + i];
        }
    }
    return SUCCESS;
}

// x = x + lr * y
// y = momentum * (lr * y), 动量法，保存用作下一轮使用
// 该方法专门为参数更新准备，因此不涉及tensor->b和tensor->n
//int addTensor(struct Tensor *x, struct Tensor *y, float lr, int n_samples, float momentum)
int addTensor(struct Tensor *x, struct Tensor *y, float lr, float momentum)
{
    CHK_NIL(x);
    CHK_NIL(y);
    CHK_ERR((x->ttype == PARAM_TENSOR_TYPE)? 0: 1);
    CHK_ERR((y->ttype == PARAM_TENSOR_TYPE)? 0: 1);
    //CHK_ERR((n_samples >0)? 0: 1);

    // STEP 0: prepare
    //lr *= -1;
    int n = y->row * y->col;
#ifdef _DEBUG
    fprintf(stdout, "(y.b, y.row, y.col, y.c, y.n) = (%d, %d, %d, %d, %d)\n", y->b, y->row, y->col, y->c, y->n);
#endif
    //fprintf(stdout, "(y.row, y.col) = (%d, %d)\n", y->row, y->col);
    // STEP 1: y = lr * y
    int i = 0;
    for (i = 0; i < n; ++i) {
        //(y->blob[i]) /= n_samples;
        (y->blob[i]) *= lr;
    }

    // STEP 2: x = x + lr
    for (i = 0; i < n; ++i) {
        x->blob[i] += y->blob[i];
    }

    // STEP 3: y = momentum * y
    for (i = 0; i < n; ++i) {
        y->blob[i] *= momentum;
    }

    return SUCCESS;
}

int softmaxTensor(struct Tensor *output, const struct Tensor *input)
{
    CHK_NIL(output);
    CHK_NIL(input);
    CHK_ERR((input->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((output->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((output->n == input->n)? 0: 1);

    int b_used = input->b_used;
    int k = output->n;

    int i;
    for (i = 0; i < b_used; ++i) {
        float largest = -FLT_MAX;
        int j;
        for(j = 0; j < k; ++j){
            if(input->blob[i * k + j] > largest) {
                largest = input->blob[i * k + j];
            }
        }
        float sum = 0;
        for(j = 0; j < k; ++j){
            float e = exp(input->blob[i * k + j] - largest);
            sum += e;
            output->blob[i * k + j] = e;
        }
        for(j = 0; j < k; ++j){
            (output->blob[i * k + j]) /= sum;
        }
    }
    output->b_used = input->b_used;
    return SUCCESS;
}

int addTensor2(struct Tensor *delta, const struct Tensor *y, const struct Tensor *gt)
{
    CHK_NIL(delta);
    CHK_NIL(y);
    CHK_NIL(gt);
    CHK_ERR((delta->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((y->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((gt->ttype == DATA_TENSOR_TYPE)? 0: 1);

    int n_elems = y->b_used * y->n;
#ifdef _DEBUG
    fprintf(stdout, "(y->b, y->row, y->col, y->c, y->n) = (%d, %d, %d, %d, %d)\n", y->b, y->row, y->col, y->c, y->n);
    fprintf(stdout, "(gt->b, gt->row, gt->col, gt->c, gt->n) = (%d, %d, %d, %d, %d)\n", gt->b, gt->row, gt->col, gt->c, gt->n);
    fprintf(stdout, "(delta->b, delta->row, delta->col, delta->c, delta->n) = (%d, %d, %d, %d, %d)\n", delta->b, delta->row, delta->col, delta->c, delta->n);
#endif
    int i;
    switch (gt->dtype) {
        case UINT8:
        for (i = 0; i < n_elems; ++i) {
            delta->blob[i] = (gt->blob_u8[i] - y->blob[i]) / (float)(gt->b_used); // 学习速率关于每个batch的样本数降低的计算，统一放在代价函数里，不需要放在每一层的update里
        }
        break;

        default:
        ERR_MSG("DType not supported yet, error.\n");
        return ERR_COD;
    }
    delta->b_used = y->b_used;

    return SUCCESS;
}

// 一个batch上的对数极大似然
int probTensor(float *val, const struct Tensor *p, const struct Tensor *gt)
{
    CHK_NIL(val);
    CHK_NIL(p);
    CHK_NIL(gt);
    CHK_ERR((gt->dtype == UINT8)? 0: 1); // one-hot表示，每个元素非0即1，因此uint8足够
    CHK_ERR((p->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((gt->ttype == DATA_TENSOR_TYPE)? 0: 1);
    CHK_ERR((p->b_used == gt->b_used)? 0: 1);

    int b_used = p->b_used;
    int k = p->n;
    float sum_log_p = 0.;
    int i, j;
    for (i = 0; i < b_used; ++i) {
        for (j = 0; j < k; ++j) {
            if (gt->blob_u8[i * k + j] != 0) {
                sum_log_p += log(p->blob[i * k + j]);
                break;
            }
        }
    }
    fprintf(stdout, "sum_log_p = %f, n = %d\n", sum_log_p, b_used);
    *val = sum_log_p / b_used; // 计算平均值, 用于观察评估寻俩效果的代价值建议与样本数无关
    return SUCCESS;
}

/*
static void log2d(const float *blob, int m, int n, FILE *fp) {
    char buf[64];
    int len;
    int i, j;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            len = snprintf(buf, 64, "%f ", blob[i * n + j]);
            fwrite(buf, sizeof(char), len, fp);
        }
        fwrite("\n", sizeof(char), 1, fp);
    }
    fwrite("\n", sizeof(char), 1, fp); // 额外增加一个空行
    fflush(fp);
}
 
static void log2dParam(const struct Tensor *t, FILE *fp)
{
    log2d(t->blob, t->row, t->col, fp);
}

static void log2dData(const struct Tensor *t, FILE * fp)
{
    log2d(t->blob, t->n, t->col, fp);
}

void logTensorParam(const struct Tensor *t)
{
    if (g_fp == NULL) return;
    log2dParam(t, g_fp);
}

void logTensorData(const struct Tensor *t)
{
    if (g_fp == NULL) return;
    log2dData(t, g_fp);
}

void logTensorStr(const char *str)
{
    if (g_fp == NULL) return;
    int fd = fileno(g_fp);
    if (write(fd, str, strlen(str) * sizeof(char)) == -1) {
        ERR_MSG("write failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
    }
    fsync(fd);
}
*/

int savetxtTensorData(const struct Tensor *t, const char *dst_dir, const char *prefix, const char *name, int n_epoch, int n_iter)
{
    CHK_NIL(t);
    CHK_NIL(dst_dir);
    CHK_NIL(prefix);
    CHK_NIL(name);
    CHK_ERR((t->ttype == DATA_TENSOR_TYPE)? 0: 1);

    char pth[NN_PATH_LEN];
    snprintf(pth, NN_PATH_LEN, "%s/epoch_%03d_iter_%03d_%s_%s_%dx%d.txt", dst_dir, n_epoch, n_iter, name, prefix, t->b_used, t->n);
    switch (t->dtype) {
        case FLOAT32:
        CHK_ERR(savetxtMatrixFlot32(pth, t->blob, t->b_used, t->n));
        break;

        case UINT8:
        CHK_ERR(savetxtMatrixUint8(pth, t->blob_u8, t->b_used, t->n));
        break;

        default:
        ERR_MSG("Tensor dtype: %s not supported yet, error.\n", getTensorDtypeStrFromEnum(t->dtype));
        return ERR_COD;
    }

    return SUCCESS;
}

int savetxtTensorParam(const struct Tensor *t, const char *dst_dir, const char *prefix, const char *name, int n_epoch, int n_iter)
{
    CHK_NIL(t);
    CHK_NIL(dst_dir);
    CHK_NIL(prefix);
    CHK_NIL(name);
    CHK_ERR((t->ttype == PARAM_TENSOR_TYPE)? 0: 1);

    char pth[NN_PATH_LEN];
    snprintf(pth, NN_PATH_LEN, "%s/epoch_%03d_iter_%03d_%s_%s_%dx%d.txt", dst_dir, n_epoch, n_iter, name, prefix, t->row, t->col);
    
    switch (t->dtype) {
        case FLOAT32:
        CHK_ERR(savetxtMatrixFlot32(pth, t->blob, t->row, t->col));
        break;

        case UINT8:
        CHK_ERR(savetxtMatrixUint8(pth, t->blob_u8, t->row, t->col));
        break;

        default:
        ERR_MSG("Tensor dtype: %s not supported yet, error.\n", getTensorDtypeStrFromEnum(t->dtype));
        return ERR_COD;
    }
    return SUCCESS;
}
