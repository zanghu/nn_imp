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

struct Tensor {
    enum DType dtype;
    int b; // batch_size
    int row;
    int col;
    int c; // channel
    int n; // 实际装填的样本数， n<=b
    float *data;
    double *data_f64;
    int *data_i32;
    long long *data_i64;
    unsigned char *data_u8;
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

void initTensorParameterAsWeight(struct Tensor *tensor)
{
    int inputs = tensor->row;
    float scale = sqrt(2. / inputs);
    int i = 0;
    for(i = 0; i < tensor->b * tensor->row * tensor->col * tensor->c; ++i){
        tensor->data[i] = scale * rand_uniform(-1, 1);
    }
}

void initTensorParameterAsBias(struct Tensor *tensor)
{
    memset(tensor->data, 0, tensor->b * tensor->row * tensor->col * tensor->c * sizeof(float));
}

int createTensor(struct Tensor **t, int batch_size, int row, int col, int channel)
{
    CHK_NIL(t);
    CHK_ERR((batch_size > 0)? 0: 1);
    CHK_ERR((row > 0)? 0: 1);
    CHK_ERR((col > 0)? 0: 1);
    CHK_ERR((channel > 0)? 0: 1);

    struct Tensor *tensor = calloc(1, sizeof(struct Tensor));
    if (!tensor) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    tensor->data = calloc(batch_size * row * col * channel, sizeof(float));
    if (tensor->data == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        free(tensor);
        return ERR_COD;
    }
    tensor->b = batch_size;
    tensor->dtype = FLOAT32;
    tensor->row = row;
    tensor->col = col;
    tensor->c = channel;
    tensor->n = 0; // 初始样本数为0

    *t = tensor;
    return SUCCESS;
}

int createTensorU8(struct Tensor **t, int batch_size, int row, int col, int channel)
{
    CHK_NIL(t);
    CHK_ERR((batch_size > 0)? 0: 1);
    CHK_ERR((row > 0)? 0: 1);
    CHK_ERR((col > 0)? 0: 1);
    CHK_ERR((channel > 0)? 0: 1);

    struct Tensor *tensor = calloc(1, sizeof(struct Tensor));
    if (!tensor) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    tensor->data_i32 = calloc(batch_size * row * col * channel, sizeof(int));
    if (tensor->data_i32 == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        free(tensor);
        return ERR_COD;
    }
    tensor->b = batch_size;
    tensor->dtype = UINT8;
    tensor->row = row;
    tensor->col = col;
    tensor->c = channel;
    tensor->n = 0; // 初始样本数为0

    *t = tensor;
    return SUCCESS;
}

int createTensorWithDataRef(struct Tensor **t, int n_samples, int row, int col, int c, const void *data, enum DType dtype)
{
    CHK_NIL(t);
    CHK_ERR((row > 0)? 0: 1);
    CHK_ERR((col > 0)? 0: 1);
    CHK_ERR((c > 0)? 0: 1);
    CHK_NIL(data);
    CHK_ERR((n_samples > 0)? 0: 1);

    struct Tensor *tensor = calloc(1, sizeof(struct Tensor));
    if (!tensor) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    switch (dtype) {
        case FLOAT32:
        tensor->data = (float *)data;
        break;

        case UINT8:
        tensor->data_u8 = (unsigned char *)data;
        break;

        default:
        ERR_MSG("DType: %s is not supported yet, error.\n", getTensorDtypeStrFromEnum(dtype));
        free(tensor);
        return ERR_COD;
    }
    tensor->dtype = dtype;
    tensor->b = n_samples;
    tensor->row = row;
    tensor->col = col;
    tensor->c = c;
    tensor->n = n_samples;

    *t = tensor;
    return SUCCESS;
}

void destroyTensor(struct Tensor *tensor)
{
    if (tensor) {
        free(tensor->data);
        free(tensor->data_f64);
        free(tensor->data_i32);
        free(tensor->data_i64);
        free(tensor->data_u8);
    }
    free(tensor);
}

int copyTensorData(void *dst, enum DType dtype, const struct Tensor *tensor)
{
    CHK_NIL(dst);
    CHK_NIL(tensor);
    CHK_ERR((tensor->dtype == dtype)? 0: 1);

    int n_elem = tensor->n * tensor->row * tensor->col * tensor->c;
    switch (dtype) {
        case FLOAT32:
        memcpy(dst, tensor->data, n_elem * sizeof(float));
        break;

        case FLOAT64:
        memcpy(dst, tensor->data_f64, n_elem * sizeof(double));
        break;

        case INT32:
        memcpy(dst, tensor->data_i32, n_elem * sizeof(int));
        break;

        case INT64:
        memcpy(dst, tensor->data_i64, n_elem * sizeof(long long));
        break;

        case UINT8:
        memcpy(dst, tensor->data_u8, n_elem * sizeof(unsigned char));
        break;

        default:
        ERR_MSG("Unknow DType: %s, error.\n", getTensorDtypeStrFromEnum(dtype));
        return ERR_COD;
    }
    return SUCCESS;
}

int getTensorShape(int *b, int *row, int *col, int *c, const struct Tensor *tensor)
{
    CHK_NIL(b);
    CHK_NIL(row);
    CHK_NIL(col);
    CHK_NIL(c);
    CHK_NIL(tensor);

    *b = tensor->b;
    *row = tensor->row;
    *col = tensor->col;
    *c = tensor->c;

    return SUCCESS;
}

int getTensorRowAndCol(int *row, int *col, const struct Tensor *tensor)
{
    CHK_NIL(row);
    CHK_NIL(col);
    CHK_NIL(tensor);
    *row = tensor->row;
    *col = tensor->col;
    return SUCCESS;
}

int getTensorBatchAndChannel(int *b, int *c, const struct Tensor *tensor)
{
    CHK_NIL(b);
    CHK_NIL(c);
    CHK_NIL(tensor);
    *b = tensor->b;
    *c = tensor->c;
    return SUCCESS;
}

int getTensorRow(int *row, const struct Tensor *tensor)
{
    CHK_NIL(row);
    CHK_NIL(tensor);
    *row = tensor->row;
    return SUCCESS;
}

int getTensorCol(int *col, const struct Tensor *tensor)
{
    CHK_NIL(col);
    CHK_NIL(tensor);
    *col = tensor->col;
    return SUCCESS;
}

int getTensorChannel(int *c, const struct Tensor *tensor)
{
    CHK_NIL(c);
    CHK_NIL(tensor);
    *c = tensor->c;
    return SUCCESS;
}

int getTensorBatch(int *b, const struct Tensor *tensor)
{
    CHK_NIL(b);
    CHK_NIL(tensor);
    *b = tensor->b;
    return SUCCESS;
}

int getTensorSamples(int *n, const struct Tensor *tensor)
{
    CHK_NIL(n);
    CHK_NIL(tensor);
    *n = tensor->n;
    return SUCCESS;
}

int getTensorDType(enum DType *dtype, const struct Tensor *tensor)
{
    CHK_NIL(dtype);
    CHK_NIL(tensor);
    *dtype = tensor->dtype;
    return SUCCESS;
}

int getTensorData(void **data, struct Tensor *tensor)
{
    CHK_NIL(data);
    CHK_NIL(tensor);

    switch (tensor->dtype) {
        case FLOAT32:
        *data = tensor->data;
        break;

        case FLOAT64:
        *data = tensor->data_f64;
        break;

        case INT32:
        *data = tensor->data_i32;
        break;

        case INT64:
        *data = tensor->data_i64;
        break;

        case UINT8:
        *data = tensor->data_u8;
        break;

        default:
        ERR_MSG("Unknow DType: %s, error.\n", getTensorDtypeStrFromEnum(tensor->dtype));
        return ERR_COD;
    }
    return SUCCESS;
}

int setTensorData(struct Tensor *tensor, const void *data, enum DType dtype, int n)
{
    CHK_NIL(tensor); 
    CHK_NIL(data);

    int n_features = tensor->col * tensor->row * tensor->c;
    switch (tensor->dtype) {
        case FLOAT32:
        switch (dtype) {
            case UINT8:
            {
                const unsigned char *src = data;
                int i, j;
                for (i = 0; i < n; ++i) {
                    for (j = 0; j < n_features; ++j) {
                        tensor->data[i * n_features + j] = (float)(src[i * n_features + j]);
                    }
                }
            }
            break;

            default:
            ERR_MSG("src_dtype: %s to dst_dtype: %s not supported, error.\n", getTensorDtypeStrFromEnum(dtype), getTensorDtypeStrFromEnum(tensor->dtype));
            return ERR_COD;
        }
        break;

        default:
        return ERR_COD;
    }
    tensor->n = n;
    return SUCCESS;
}

int setTensorBatchAndDataByReplace(struct Tensor *tensor, const void *data, int n_samples, enum DType dtype, int need_free)
{
    CHK_NIL(tensor);
    CHK_ERR((n_samples > 0)? 0: 1);
    CHK_NIL(data);
    CHK_ERR((tensor->dtype == dtype)? 0: 1);
    switch (dtype) {
        case UINT8:
        if (need_free) {
            free(tensor->data_u8);
        }
        tensor->data_u8 = (unsigned char *)data;
        break;

        case FLOAT32:
        if (need_free) {
            free(tensor->data);
        }
        tensor->data = (float *)data;
        break;

        default:
        ERR_MSG("DType: %s is not supported yet, error.\n", getTensorDtypeStrFromEnum(dtype));
        return ERR_COD;
    }
    tensor->b = n_samples;
    tensor->n = n_samples;
    return SUCCESS;
}

// 前向传播过程的非线性运算部分
int activateTensor(struct Tensor *y, const struct Tensor *x, enum ActivationType act_type)
{
    CHK_NIL(x);
    CHK_ERR((x->n > 0)? 0: 1);

    int i = 0;
    switch (act_type) {
        case LOGISTIC:
        for (i = 0; i < x->n * x->row * x->col * x->c; ++i) {
            //y->data[i] = 1./(1. + exp(-1 * (x->data[i])));
            y->data[i] = logistic_activate(x->data[i]);
        }
        break;

        default:
        ERR_MSG("Unknow activation_type, error.\n");
        return ERR_COD;
    }
    y->n = x->n;
    return SUCCESS;
}

// 反向传播过程的非线性运算部分, 计算后的结果会替换输入值delta的原有内容
int deactivateTensor(struct Tensor *delta_out, const struct Tensor *delta_in, const struct Tensor *input, enum ActivationType act_type)
{
    CHK_NIL(delta_out);
    CHK_NIL(delta_in);
    CHK_NIL(input);
    CHK_ERR((delta_in->n > 0)? 0: 1);
    CHK_ERR((delta_in->n == input->n)? 0: 1);

    int i = 0;
    switch (act_type) {
        case LOGISTIC:
        for (i = 0; i < input->n * input->row * input->col * input->c; ++i) {
            delta_out->data[i] = logistic_gradient(input->data[i]) * delta_in->data[i]; // 注意这里的x应该是outputs[i]，而不是hiddens[i]
        }
        break;

        default:
        ERR_MSG("Unknow dectivation_type, error.\n");
        return ERR_COD;
    }
    delta_out->n = delta_in->n;
    return SUCCESS;
}

/**
 * @param x: 输入参数，shape = (batch_size, n_input)
 * @param y: 输入参数，shape = (barch_size, n_output)
 * @param z: 输出参数, shape = (n_input, n_output)
 * @param b: 输入参数, shape = (1, n_output)
 */
int linearTensor(struct Tensor *z, const struct Tensor *x, int xt, const struct Tensor *y, int yt, const struct Tensor *b)
{
    CHK_NIL(z);
    CHK_NIL(x);
    CHK_NIL(y);
    CHK_ERR((x->n > 0)? 0: 1);
#ifdef _DEBUG
    fprintf(stdout, "(z.b, z.row, z.col, z.c, z.n) = (%d, %d, %d, %d, %d)\n", z->b, z->row, z->col, z->c, z->n);
    fprintf(stdout, "(y.b, y.row, y.col, y.c, y.n) = (%d, %d, %d, %d, %d)\n", y->b, y->row, y->col, y->c, y->n);
    fprintf(stdout, "(x.b, x.row, x.col, x.c, x.n) = (%d, %d, %d, %d, %d)\n", x->b, x->row, x->col, x->c, x->n);
#endif
    //CHK_ERR((z->col == y->row)? 0: 1);
    //CHK_ERR((x->col == y->col)? 0: 1);
    //CHK_ERR((b->col == y->row)? 0: 1);

    if (b) {
        int i = 0;
        for (i = 0; i < x->n; ++i) {
            memcpy(z->data + i * z->col, b->data, sizeof(float) * b->col);
        }
    }

    // gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
    //     float *A, int lda, 
    //     float *B, int ldb,
    //     float BETA,
    //     float *C, int ldc)
    gemm(xt, yt, x->n, y->row, x->col, 1., 
        x->data, x->col, 
        y->data, y->row, 
        1., 
        z->data, y->row);
    z->n = x->n;
    return SUCCESS;
}

int linearTensor1(struct Tensor *z, const struct Tensor *x, int xt, const struct Tensor *y, int yt, const struct Tensor *b)
{
    CHK_NIL(z);
    CHK_NIL(x);
    CHK_NIL(y);
    CHK_ERR((x->n > 0)? 0: 1);
#ifdef _DEBUG
    fprintf(stdout, "(z.b, z.row, z.col, z.c, z.n) = (%d, %d, %d, %d, %d)\n", z->b, z->row, z->col, z->c, z->n);
    fprintf(stdout, "(y.b, y.row, y.col, y.c, y.n) = (%d, %d, %d, %d, %d)\n", y->b, y->row, y->col, y->c, y->n);
    fprintf(stdout, "(x.b, x.row, x.col, x.c, x.n) = (%d, %d, %d, %d, %d)\n", x->b, x->row, x->col, x->c, x->n);
#endif
    //CHK_ERR((z->col == y->row)? 0: 1);
    //CHK_ERR((x->col == y->col)? 0: 1);
    //CHK_ERR((b->col == y->row)? 0: 1);

    if (b) {
        int i = 0;
        for (i = 0; i < x->n; ++i) {
            memcpy(z->data + i * z->col, b->data, sizeof(float) * b->col);
        }
    }

    // gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
    //     float *A, int lda, 
    //     float *B, int ldb,
    //     float BETA,
    //     float *C, int ldc)
    gemm(xt, yt, x->n, y->col, y->row, 1., 
        x->data, x->col, 
        y->data, y->col, 
        1., 
        z->data, z->col);
    z->n = x->n;
    return SUCCESS;
}

int linearTensor2(struct Tensor *z, const struct Tensor *x, int xt, const struct Tensor *y, int yt, const struct Tensor *b)
{
    CHK_NIL(z);
    CHK_NIL(x);
    CHK_NIL(y);
    CHK_ERR((x->n > 0)? 0: 1);
#ifdef _DEBUG
    fprintf(stdout, "(z.b, z.row, z.col, z.c, z.n) = (%d, %d, %d, %d, %d)\n", z->b, z->row, z->col, z->c, z->n);
    fprintf(stdout, "(y.b, y.row, y.col, y.c, y.n) = (%d, %d, %d, %d, %d)\n", y->b, y->row, y->col, y->c, y->n);
    fprintf(stdout, "(x.b, x.row, x.col, x.c, x.n) = (%d, %d, %d, %d, %d)\n", x->b, x->row, x->col, x->c, x->n);
#endif
    //CHK_ERR((z->col == y->row)? 0: 1);
    //CHK_ERR((x->col == y->col)? 0: 1);
    //CHK_ERR((b->col == y->row)? 0: 1);

    if (b) {
        int i = 0;
        for (i = 0; i < x->n; ++i) {
            memcpy(z->data + i * z->col, b->data, sizeof(float) * b->col);
        }
    }

    // gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
    //     float *A, int lda, 
    //     float *B, int ldb,
    //     float BETA,
    //     float *C, int ldc)
    gemm(xt, yt, x->col, y->col, x->n, 1., 
        x->data, x->row, 
        y->data, y->col, 
        1., 
        z->data, z->col);
    //z->n = x->n;
    return SUCCESS;
}


// 2个形状完全相同的矩阵x和y的元素做Pointwise乘法, 之后每行求和, 压缩成一个向量, 结果保存在z中
// 该函数用来在反向传播时计算偏置bias的梯度
//int mulTensorPointwiseAndSum(struct Tensor *z, const struct Tensor *x, const struct Tensor *y)
int sumTensorAxisCol(struct Tensor *z, const struct Tensor *x)
{
    CHK_NIL(z);
    CHK_NIL(x);
#ifdef _DEBUG
    fprintf(stdout, "(x.b, x.row, x.col, x.c, x.n) = (%d, %d, %d, %d, %d)\n", x->b, x->row, x->col, x->c, x->n);
    fprintf(stdout, "(z.b, z.row, z.col, z.c. z.n) = (%d, %d, %d, %d, %d)\n", z->b, z->row, z->col, z->c, z->n);
#endif
    CHK_ERR((x->col == z->col)? 0: 1);

    int n = x->n; // batch_size
    int c = x->col; // n_output
    int i, j;
    for (i = 0; i < c; ++i) {
        z->data[i] = 0.;
        for (j = 0; j < n; ++j) {
            z->data[i] += x->data[j * c + i];
        }
    }
    return SUCCESS;
}

// x = x + lr * y
// y = momentum * (lr * y), 动量法，保存用作下一轮使用
// 该方法专门为参数更新准备，因此不涉及tensor->b和tensor->n
int addTensor(struct Tensor *x, struct Tensor *y, float lr, float momentum)
{
    CHK_NIL(x);
    CHK_NIL(y);

    // STEP 0: prepare
    //lr *= -1;
    int n = y->row * y->col * y->c;
#ifdef _DEBUG
    fprintf(stdout, "(y.b, y.row, y.col, y.c, y.n) = (%d, %d, %d, %d, %d)\n", y->b, y->row, y->col, y->c, y->n);
#endif
    // STEP 1: y = lr * y
    int i = 0;
    for (i = 0; i < n; ++i) {
        (y->data[i]) *= lr;
    }

    // STEP 2: x = x + lr
    for (i = 0; i < n; ++i) {
        x->data[i] += y->data[i];
    }

    // STEP 3: y = momentum * y
    for (i = 0; i < n; ++i) {
        y->data[i] *= momentum;
    }

    return SUCCESS;
}

int softmaxTensor(struct Tensor *output, const struct Tensor *input)
{
    CHK_NIL(output);
    CHK_NIL(input);

    int n = input->n;
    int k = output->col;

    int i;
    for (i = 0; i < n; ++i) {
        float largest = -FLT_MAX;
        int j;
        for(j = 0; j < k; ++j){
            if(input->data[i * k + j] > largest) {
                largest = input->data[i * k + j];
            }
        }
        float sum = 0;
        for(j = 0; j < k; ++j){
            float e = exp(input->data[i * k + j] - largest);
            sum += e;
            output->data[i * k + j] = e;
        }
        for(j = 0; j < k; ++j){
            (output->data[i * k + j]) /= sum;
        }
    }
    output->n = input->n;
    return SUCCESS;
}

int addTensor2(struct Tensor *delta, const struct Tensor *y, const struct Tensor *gt)
{
    CHK_NIL(delta);
    CHK_NIL(y);
    CHK_NIL(gt);

    int n = y->n * y->col;
#ifdef _DEBUG
    fprintf(stdout, "(y->b, y->row, y->col, y->c, y->n) = (%d, %d, %d, %d, %d)\n", y->b, y->row, y->col, y->c, y->n);
    fprintf(stdout, "(gt->b, gt->row, gt->col, gt->c, gt->n) = (%d, %d, %d, %d, %d)\n", gt->b, gt->row, gt->col, gt->c, gt->n);
    fprintf(stdout, "(delta->b, delta->row, delta->col, delta->c, delta->n) = (%d, %d, %d, %d, %d)\n", delta->b, delta->row, delta->col, delta->c, delta->n);
#endif
    int i;
    switch (gt->dtype) {
        case UINT8:
        for (i = 0; i < n; ++i) {
            delta->data[i] = gt->data_u8[i] - y->data[i];
        }
        break;

        default:
        ERR_MSG("DType not supported yet, error.\n");
        return ERR_COD;
    }
    delta->n = y->n;

    return SUCCESS;
}

// 一个batch上的对数极大似然
int probTensor(float *val, const struct Tensor *p, const struct Tensor *gt)
{
    CHK_NIL(val);
    CHK_NIL(p);
    CHK_NIL(gt);
    CHK_ERR((gt->dtype == UINT8)? 0: 1); // one-hot表示，每个元素非0即1，因此uint8足够

    int n = p->n;
    int c = p->col;
    float sum_log_p = 0.;
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < c; ++j) {
            if (gt->data_u8[i * c + j] != 0) {
                //sum_log_p += p->data[i * c + j];
                sum_log_p += log(p->data[i * c + j]);
                break;
            }
        }
    }
    fprintf(stdout, "sum_log_p = %f, n = %d\n", sum_log_p, n);
    *val = sum_log_p / n; // 计算平均值, 用于观察评估寻俩效果的代价值建议与样本数无关
    return SUCCESS;
}

static void log2d(const float *data, int m, int n, FILE *fp) {
    char buf[64];
    int len;
    int i, j;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            len = snprintf(buf, 64, "%f ", data[i * n + j]);
            fwrite(buf, sizeof(char), len, fp);
        }
        fwrite("\n", sizeof(char), 1, fp);
    }
    fwrite("\n", sizeof(char), 1, fp); // 额外增加一个空行
    fflush(fp);
}
 
static void log2dParam(const struct Tensor *t, FILE *fp)
{
    log2d(t->data, t->row, t->col, fp);
}

static void log2dData(const struct Tensor *t, FILE * fp)
{
    log2d(t->data, t->n, t->col, fp);
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
